import seaborn as sns

import ydnpd


class PrivacyUtilityTradeoffTask:
    @staticmethod
    def execute(self):
        raise RuntimeError("Use HyperParamSearchTask to produce results")

    def evaluate(self):
        raise RuntimeError("Use HyperParamSearchTask to evaluate")

    @staticmethod
    def plot(hparam_results, experiments, metric=None):
        evaluation_df = ydnpd.HyperParamSearchTask.evaluate(
            hparam_results, ydnpd.config.EXPERIMENTS, metric
        ).reset_index()

        evaluation_df["correspond_test"] *= 100

        g = sns.relplot(
            data=evaluation_df,
            x="epsilon",
            y="correspond_test",
            hue="experiment",
            kind="line",
            col="synth_name",
        )

        def format_tick(x):
            return f"{x:.2f}" if not float(x).is_integer() else f"{int(x)}"

        epsilons = evaluation_df["epsilon"].unique()
        g.set(
            xticks=epsilons, xticklabels=[format_tick(epsilon) for epsilon in epsilons]
        )

        return g
