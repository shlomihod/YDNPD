import itertools as it
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import wandb

from ydnpd.dataset import load_dataset
from ydnpd.synthesis import generate_synthetic_data
from ydnpd.evaluation import evaluate_two
from ydnpd.tasks import DPTask
from ydnpd.utils import _freeze


RANDOM_STATE_TRAIN_TEST_SPLIT = 42


class UtilityTask(DPTask):

    METRIC_DEFAULT = "marginals_3_max_abs_diff_error"

    def __init__(
        self,
        dataset_name: str,
        epsilons: list[float],
        synth_name: str,
        hparam_dims: dict[str, list],
        num_runs: int,
        eval_split_proportion: float,
        verbose: bool = True,
        with_wandb: bool = False,
        wandb_kwargs: dict = None,
        evaluation_kwargs: dict = None,
    ):

        if not 0 < eval_split_proportion < 1:
            raise ValueError(
                "`eval_split_proportion` must be float number in the range (0, 1)"
            )

        self.dataset_name = dataset_name
        self.epsilons = epsilons
        self.synth_name = synth_name
        self.hparam_dims = hparam_dims
        self.num_runs = num_runs
        self.eval_split_proportion = eval_split_proportion
        self.verbose = verbose
        self.with_wandb = with_wandb
        self.evaluation_kwargs = (
            evaluation_kwargs if evaluation_kwargs is not None else {}
        )

        if not with_wandb:
            if wandb_kwargs is not None:
                raise ValueError("`wandb_kwargs` must be None if `with_wandb` is False")
        else:
            self.wandb_kwargs = wandb_kwargs if wandb_kwargs is not None else {}

        self.hparam_space = [
            dict(zip(hparam_dims, values))
            for values in it.product(*hparam_dims.values())
        ]

    def __str__(self):
        return f"<UtilityTask (#configs={self.size()}): {self.synth_name} & {self.dataset_name}>"

    def size(self) -> int:
        return len(self.epsilons) * len(self.hparam_space) * self.num_runs

    def _execute_run(
        self,
        train_dataset: pd.DataFrame,
        eval_dataset: pd.DataFrame,
        schema: dict,
        epsilon: float,
        hparams: dict,
    ) -> tuple[dict, pd.DataFrame]:
        synth_dataset = generate_synthetic_data(
            train_dataset, schema, epsilon, self.synth_name, **hparams
        )

        metric_results = evaluate_two(
            train_dataset, eval_dataset, synth_dataset, schema, **self.evaluation_kwargs
        )

        return metric_results, synth_dataset

    def execute(self) -> list[dict]:

        dataset, schema = load_dataset(self.dataset_name)

        train_dataset, eval_dataset = train_test_split(
            dataset,
            test_size=self.eval_split_proportion,
            random_state=RANDOM_STATE_TRAIN_TEST_SPLIT,
        )

        results = []

        for epsilon in self.epsilons:

            for hparams in self.hparam_space:

                config = {
                    "dataset_name": self.dataset_name,
                    "epsilon": epsilon,
                    "synth_name": self.synth_name,
                    "hparams": hparams,
                }

                if self.with_wandb:
                    wandb.init(project="ydnpd", config=config, **self.wandb_kwargs)

                for run_id in range(self.num_runs):

                    if self.verbose:
                        print(
                            f"{self.__class__.__name__}: dataset = {self.dataset_name} synth_name={self.synth_name}, epsilon={epsilon}, hparams={hparams} run={run_id + 1}/{self.num_runs}"
                        )

                    try:
                        (metric_results, synth_dataset) = self._execute_run(
                            train_dataset, eval_dataset, schema, epsilon, hparams
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        if self.with_wandb:
                            wandb.log({"_error": str(e)})

                        metric_results = {}
                        synth_dataset = None
                    else:
                        if self.with_wandb:
                            wandb.log(metric_results)

                    results.append(
                        config
                        | {"evaluation": metric_results, "synth_dataset": synth_dataset}
                    )

                if self.with_wandb:
                    wandb.finish()

        return results

    @staticmethod
    def evaluate(hparam_results, experiemnts, metric=None):
        if metric is None:
            metric = UtilityTask.METRIC_DEFAULT
        metric_column = f"evaluation_{metric}"

        results_df = UtilityTask.process(hparam_results)

        best_hparams_df = results_df.groupby(["dataset_name", "synth_name", "epsilon"])[
            metric_column
        ].idxmin()

        def extractor(test_name, dev_name):

            def function(r):
                metric_column = f"evaluation_{metric}"
                synth_name, epsilon = r.name
                hparams = results_df.iloc[r.item()]["hparams_frozen"]
                base_mask = (results_df["synth_name"] == synth_name) & (
                    results_df["epsilon"] == epsilon
                )
                best_dev_result = results_df.loc[
                    (results_df["dataset_name"] == dev_name)
                    & (results_df["hparams_frozen"] == hparams)
                    & base_mask,
                    metric_column,
                ].item()
                correspond_test_result = results_df.loc[
                    (results_df["dataset_name"] == test_name)
                    & (results_df["hparams_frozen"] == hparams)
                    & base_mask,
                    metric_column,
                ].item()
                test_results = results_df.loc[
                    (results_df["dataset_name"] == test_name) & base_mask, metric_column
                ]
                return {
                    "quantile": (correspond_test_result > test_results).sum()
                    / len(test_results),
                    "best_dev": best_dev_result,
                    "correspond_test": correspond_test_result,
                    "best_test": test_results.min(),
                    "median_test": test_results.median(),
                    "worst_test": test_results.max(),
                }

            return function

        hparams_evaluation_df = (
            pd.concat(
                [
                    (
                        pd.DataFrame(best_hparams_df[dev_name])
                        .apply(
                            extractor(experiemnts.test_name, dev_name),
                            result_type="expand",
                            axis=1,
                        )
                        .reset_index()
                        .assign(
                            dev_name=dev_name,
                            test_name=experiemnts.test_name,
                            experiment=f"{experiemnts.test_name}/{dev_name}",
                        )
                    )
                    for dev_name in experiemnts.dev_names
                ]
            )
            .set_index(["synth_name", "experiment", "epsilon"])
            .sort_index()
            .drop(columns=["test_name", "dev_name"])
        )

        return hparams_evaluation_df

    @staticmethod
    def plot(hparam_results, experiments, metric=None):

        if metric is None:
            metric = UtilityTask.METRIC_DEFAULT
        metric_column = f"evaluation_{metric}"

        results_df = UtilityTask.process(hparam_results)

        evaluation_df = UtilityTask.evaluate(
            hparam_results, experiments, metric
        ).reset_index()

        results_df[metric_column] *= 100
        value_columns = ["best_dev", "best_test", "correspond_test"]
        evaluation_df[value_columns] *= 100

        epsilons = results_df["epsilon"].unique()

        def plot_dev_vs_test():
            def expender(test_name, dev_name):
                def function(g):
                    g = g[["dataset_name", "hparams_frozen", metric_column]].rename(
                        columns={metric_column: "metric"}
                    )
                    df = (
                        pd.merge(
                            g.query("dataset_name == @test_name"),
                            g.query("dataset_name == @dev_name"),
                            on="hparams_frozen",
                            suffixes=("_test", "_dev"),
                        )
                        .rename(
                            columns={
                                "dataset_name_test": "test_name",
                                "dataset_name_dev": "dev_name",
                            }
                        )
                        .assign(
                            experiment=f"{test_name}/{dev_name}",
                        )
                    )
                    return df

                return function

            experiment_df = pd.concat(
                [
                    results_df.groupby(["synth_name", "epsilon"])
                    .apply(
                        expender(experiments.test_name, dev_name), include_groups=False
                    )
                    .reset_index()
                    for dev_name in experiments.dev_names
                ]
            )

            g = sns.lmplot(
                data=experiment_df,
                x="metric_dev",
                y="metric_test",
                hue="epsilon",
                row="synth_name",
                col="experiment",
                ci=None,
            )

            # g.set(xlim=(0, 100), ylim=(0, 100))

            return g

        def plot_dev_within_test():
            def plot_swarm_and_line(data, **kwargs):

                synth_name = data["synth_name"].unique().item()

                sns.swarmplot(
                    data=data, x="epsilon", y=metric_column, color="black", alpha=0.6
                )

                sns.pointplot(
                    data=evaluation_df.query(f"synth_name == '{synth_name}'"),
                    x="epsilon",
                    y="correspond_test",
                    hue="experiment",
                    marker="x",
                )

            g = sns.FacetGrid(
                results_df[results_df["dataset_name"] == experiments.test_name],
                col="synth_name",
                height=6,
                aspect=1.5,
            )
            g.map_dataframe(plot_swarm_and_line)

            g.add_legend()

            return g

        def plot_best_dev():

            best_hparams_df = (
                results_df.groupby(["dataset_name", "synth_name", "epsilon"])[
                    metric_column
                ]
                .mean()
                .reset_index()
            )

            g = sns.relplot(
                data=best_hparams_df,
                x="epsilon",
                y=metric_column,
                hue="dataset_name",
                col="synth_name",
                kind="line",
                errorbar=None,
            )

            g.set(xticks=epsilons)

            return g

        def plot_best_dev_vs_test():

            melted_df = evaluation_df.melt(
                id_vars=["synth_name", "experiment", "epsilon"],
                value_vars=["best_dev", "best_test", "correspond_test"],
                var_name="metric",
                value_name="value",
            )

            g = sns.relplot(
                data=melted_df,
                x="epsilon",
                y="value",
                hue="metric",
                col="experiment",
                row="synth_name",
                kind="line",
                errorbar=None,
            )

            g.set(xticks=epsilons)

            return g

        return (
            plot_dev_vs_test(),
            plot_dev_within_test(),
            plot_best_dev(),
            plot_best_dev_vs_test(),
        )

    @staticmethod
    def process(hparam_results):
        df = pd.DataFrame(hparam_results)

        df["hparams_frozen"] = df["hparams"].apply(_freeze)

        metric_columns = []
        for metric in list(df.loc[0, "evaluation"].keys()):
            if not metric.startswith("_"):
                metric_column = f"evaluation_{metric}"
                df[metric_column] = df["evaluation"].apply(lambda x: x[metric])
                metric_columns.append(metric_column)

        df = (
            df.groupby(["dataset_name", "synth_name", "epsilon", "hparams_frozen"])[
                metric_columns
            ]
            .mean()
            .reset_index()
        )

        return df
