import time
import itertools as it
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import wandb

from ydnpd.dataset import load_dataset
from ydnpd.synthesis import generate_synthetic_data
from ydnpd.tasks import DPTask
from ydnpd.utils import _freeze


class HyperParamSearchTask(DPTask):

    METRIC_DEFAULT = "marginals_up_to_3_max_abs_diff_errors"

    def __init__(
        self,
        dataset_name: str,
        epsilons: list[float],
        synth_name: str,
        hparam_dims: dict[str, list],
        evaluation_fn: Callable,
        num_runs: int,
        verbose: bool = True,
        with_wandb: bool = False,
        wandb_kwargs: dict = None,
    ):
        self.dataset_name = dataset_name
        self.epsilons = epsilons
        self.synth_name = synth_name
        self.hparam_dims = hparam_dims
        self.evaluation_fn = evaluation_fn
        self.num_runs = num_runs
        self.verbose = verbose
        self.with_wandb = with_wandb

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
        return f"<HyperParamSearchTask (#configs={self.size()}): {self.synth_name} & {self.dataset_name}>"

    def size(self) -> int:
        return len(self.epsilons) * len(self.hparam_space) * self.num_runs

    def execute(self) -> list[dict]:

        dataset, schema = load_dataset(self.dataset_name)

        results = []

        for epsilon in self.epsilons:

            for hparams in self.hparam_space:

                config = {
                    "dataset_name": self.dataset_name,
                    "epsilon": epsilon,
                    "synth_name": self.synth_name,
                    "hparams": hparams,
                } | {f"hparam_{k}": v for k, v in hparams.items()}

                if self.with_wandb:
                    wandb.init(project="ydnpd", config=config)

                evaluations = []
                for run_id in range(self.num_runs):

                    if self.verbose:
                        print(
                            f"{self.__class__.__name__}: dataset = {self.dataset_name} synth_name={self.synth_name}, epsilon={epsilon}, hparams={hparams} run={run_id}"
                        )

                    synth_dataset = generate_synthetic_data(
                        dataset, schema, epsilon, self.synth_name, **hparams
                    )

                    metric_results = self.evaluation_fn(dataset, synth_dataset, schema)
                    evaluations.append(metric_results)

                    if self.with_wandb:
                        wandb.log(metric_results)

                    results.append(config | {"evaluation": metric_results})

        if self.with_wandb:
            wandb.finish()

        return results

    def evaluate(self, results, *, dev_name, test_name,
                 metric=None, **kwargs) -> dict[str, float]:

        start_time = time.time()

        dev_results = results[dev_name]
        test_results = results[test_name]

        if metric is None:
            metric = self.METRIC_DEFAULT

        def get_metric(result):
            return [run[metric]
                    for run in result["evaluation"]]

        def get_metric_mean(result):
            return np.mean(get_metric(result))

        evaluation = {}

        for epsilon in self.epsilons:

            epsilon_dev_results = dev_results[str(epsilon)]
            epsilon_test_results = test_results[str(epsilon)]

            dev_test_results = zip(epsilon_dev_results, epsilon_test_results)
            dev_test_by_min_dev_result = min(dev_test_results, key=lambda x: np.mean(get_metric(x[0])))
            test_by_min_dev_metric_values = get_metric_mean(dev_test_by_min_dev_result[1])
            test_metric_values = np.array([get_metric_mean(result) for result in epsilon_test_results])

            # precentile
            evaluation[str(epsilon)] = (sum(test_by_min_dev_metric_values >= test_metric_values)
                                        / len(test_metric_values))

            # prop in top-k
            # sorted_dev_results = sorted(epsilon_dev_results, key=get_metric, reverse=True)
            # sorted_test_results = sorted(epsilon_test_results, key=get_metric, reverse=True)

            # top_k_dev_hparams = [result["hparams"] for result in sorted_dev_results][:top_k]
            # top_k_test_hparams = [result["hparams"] for result in sorted_test_results][:top_k]

            # evaluation[str(epsilon)] = sum(hparams in top_k_test_hparams
            #                                for hparams in top_k_dev_hparams) / top_k

        end_time = time.time()
        evaluation["duration"] = end_time - start_time

        return evaluation

    def plot(self, results, *, dev_name, test_name,
             metric=None, **kwargs):

        if metric is None:
            metric = self.METRIC_DEFAULT

        dev_results = results[dev_name]
        test_results = results[test_name]

        task_evaluation = self.evaluate(results,
                                        dev_name=dev_name,
                                        test_name=test_name,
                                        metric=metric)

        results_df = HyperParamSearchTask.combine_results(dev_results=dev_results,
                                                          test_results=test_results,
                                                          metric=metric)

        for role in ["dev", "test"]:
            results_df[f"metric_{role}"] *= 100

        results_df["epsilon"] = results_df["epsilon"].apply(
            lambda x: f"{x} ({100 * task_evaluation[str(x)]:.1f}%)"
        )

        g = sns.lmplot(data=results_df, x="metric_dev", y="metric_test",
                       hue="epsilon",
                       ci=None,
                       facet_kws={'legend_out': True})

        g.figure.suptitle(f"Dev vs Test {metric}")
        g.set_axis_labels(f"Dev ({dev_name})", f"Test ({test_name})")

        g._legend.set_title(f"ε (% test by min dev)")

        x_min, x_max = g.ax.get_xlim()
        y_min, y_max = g.ax.get_ylim()
        common_min = min(x_min, y_min)
        common_max = max(x_max, y_max)
        g.ax.set_xlim(common_min, common_max)
        g.ax.set_ylim(common_min, common_max)

        return g

    @staticmethod
    def combine_results(*, dev_results, test_results,
                        metric=None) -> pd.DataFrame:

        if metric is None:
            metric = HyperParamSearchTask.METRIC_DEFAULT

        def to_df(results):
            df = pd.DataFrame(sum(results.values(), []))
            df["hparams_frozen"] = df["hparams"].apply(_freeze)
            df["metric"] = df.apply(
                    lambda row: np.mean([run[metric] for run in row["evaluation"]]),
                    axis=1)
            return df

        dev_df = to_df(dev_results)
        test_df = to_df(test_results)

        results_df = pd.merge(dev_df, test_df,
                              how="outer",
                              on=("epsilon", "hparams_frozen"),
                              suffixes=("_dev", "_test"))

        return results_df
