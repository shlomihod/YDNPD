import itertools as it

import traceback

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import wandb

from ydnpd.utils import _freeze
from ydnpd.datasets.loader import load_dataset, split_train_eval_datasets
from ydnpd.harness.synthesis import generate_synthetic_data
from ydnpd.harness.evaluation import evaluate_two
from ydnpd.harness.tasks import DPTask


class UtilityTask(DPTask):

    METRIC_DEFAULT = "marginals_3_max_abs_diff_error"

    def __init__(
        self,
        dataset_pointer: str | tuple[str, str],
        epsilons: list[float],
        synth_name: str,
        hparam_dims: dict[str, list],
        num_runs: int,
        verbose: bool = True,
        with_wandb: bool = False,
        wandb_kwargs: dict = None,
        evaluation_kwargs: dict = None,
    ):

        self.dataset_pointer = dataset_pointer
        if isinstance(dataset_pointer, str):
            self.dataset_name, self.dataset_path = dataset_pointer, None
        elif isinstance(dataset_pointer, tuple):
            self.dataset_name, self.dataset_path = dataset_pointer
        else:
            raise TypeError(f"`dataset_pointer` should be either string or 2-tuple of strings")

        self.dataset_family, _ = self.dataset_name.split("/")

        self.epsilons = epsilons
        self.synth_name = synth_name
        self.hparam_dims = hparam_dims
        self.num_runs = num_runs
        self.verbose = verbose
        self.with_wandb = with_wandb
        self.evaluation_kwargs = (
            evaluation_kwargs["_"] | evaluation_kwargs[self.dataset_family]
            if evaluation_kwargs is not None else {}
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

        dataset, schema, _ = load_dataset(self.dataset_name, self.dataset_path)

        train_dataset, eval_dataset = split_train_eval_datasets(dataset)

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
                        # print(f"Error: {e}")
                        print(traceback.format_exc())



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
        plt.style.use(['science', 'no-latex'])
        
        sns.set_context("paper", rc={"axes.titlesize":16, 
                                     "axes.labelsize":14, 
                                     "xtick.labelsize":12, 
                                     "ytick.labelsize":12})

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
                height=4,
                aspect=1, 
            )

            g.set(xlim=(0, 100), ylim=(0, 100))
            g.set_titles("{row_name} | {col_name}")
            g.set_axis_labels("Dev Metric (%)", "Test Metric (%)")
            g.fig.suptitle("Dev vs Test Performance", y=1.02, fontsize=18) 

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
                height=4,
                aspect=1, 
            )
            g.map_dataframe(plot_swarm_and_line)

            g.add_legend()
            g.set_axis_labels(r"$\epsilon$", f"{metric} (%)")
            g.set_titles("{col_name}")

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
                height=4,  
                aspect=1,
            )

            g.set(xticks=epsilons)
            g.set_axis_labels(r"$\epsilon$", f"Mean {metric} (%)")
            g.set_titles("{col_name}")

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
                height=4, 
                aspect=1, 
            )

            g.set(xticks=epsilons)
            g.set_axis_labels(r"$\epsilon$", "Metric Value (%)")
            g.set_titles("{row_name} | {col_name}")
            g.fig.suptitle("Best Dev vs Test Metrics", y=1.02, fontsize=18) 

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
