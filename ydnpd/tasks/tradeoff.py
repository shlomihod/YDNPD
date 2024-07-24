import pandas as pd
import seaborn as sns

from ydnpd.tasks.hparams import HyperParamSearchTask


class PrivacyUtilityTradeoffTask:

    def __init__(self, hparams_task: HyperParamSearchTask):
        self.hparams_task = hparams_task

    def execute(self):
        raise RuntimeError("Use HyperParamSearchTask to produce results")

    def evaluate(self, hparams_results, *, dev_name, test_name, metric=None, **kwargs):

        if metric is None:
            metric = self.hparams_task.METRIC_DEFAULT

        dev_results = hparams_results[dev_name]
        test_results = hparams_results[test_name]

        dev_results = hparams_results[dev_name]
        test_results = hparams_results[test_name]

        results_df = HyperParamSearchTask.combine_results(dev_results=dev_results,
                                                          test_results=test_results)
        evaluation_df = (results_df
                         .groupby("epsilon")
                         .apply(lambda g:
                                pd.Series({f"{take}_{by}": g.at[g[f"evaluation_{by}"].str[metric].idxmin(),
                                                                f"evaluation_{take}"][metric]
                                                                for take, by in [("dev", "dev"),
                                                                                 ("test", "dev"),
                                                                                 ("test", "test")]}))
                         )

        return evaluation_df

    def plot(self, hparams_results, *, dev_name, test_name, metric=None, **kwargs):
        if metric is None:
            metric = self.hparams_task.METRIC_DEFAULT

        # TODO: refactor
        num_records = {}
        for dataset_role in ["dev", "test"]:
            dataset_name = {"dev": dev_name, "test": test_name}[dataset_role]
            num_records[dataset_role] = len(list(hparams_results[dataset_name].values())[0][0]["synth_dataset"])

        evaluation_df = self.evaluate(hparams_results,
                                      dev_name=dev_name,
                                      test_name=test_name)

        evaluation_df = evaluation_df.apply(lambda x: x / num_records[x.name.split("_")[0]]).multiply(100)

        melted_df = (evaluation_df
                     .reset_index()
                     .melt(id_vars=["epsilon"],
                           var_name="condition",
                           value_name="value")
                     )

        g = sns.relplot(data=melted_df,
                        x="epsilon", y="value", hue="condition",
                        kind="line")

        g.figure.suptitle(f"P-U Tradeoff of Dev ({dev_name}) and Test ({test_name})")
        g.set(xticks=melted_df["epsilon"].unique())
        g.set_axis_labels("ε", metric)

        return g
