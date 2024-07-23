from ydnpd.tasks.hparams import HyperParamSearchTask



class PrivacyUtilityTradeoffTask:

    def __init__(self):
        pass

    def execute(self):
        raise RuntimeError("Use HyperParamSearchTask to produce results")

    def evaluate(self, *, hparams_dev_results, hparams_test_results, **kwargs) -> dict[str, float]:
        results_df = HyperParamSearchTask.combine_results(dev_results=hparams_dev_results,
                                                            test_results=hparams_test_results)
        
        pass
        # pick the best configuration per epsilon in dev

    def plot(self, *, dev_results, test_results, **kwargs):
        pass


