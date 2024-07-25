from functools import partial

from ydnpd import load_dataset, evaluate_two, HyperParamSearchTask
import ydnpd.config


def run_hparam_task(synth_name, dataset_name):

    evaluation_fn = partial(evaluate_two,
                            classification_target_column=ydnpd.config.CLASSIFICATION_TARGET_COLUMN,
                            classification_split_proportion=ydnpd.config.CLASSIFICATION_SPLIT_PROPORTION,
                            marginals_up_to_k=ydnpd.config.MARGINALS_UP_TO_K)

    task = HyperParamSearchTask(epsilons=ydnpd.config.EPSILONS,
                            synth_name=synth_name,
                            hparam_dims=ydnpd.config.HPARAMS_DIMS[synth_name],
                            evaluation_fn=evaluation_fn,
                            num_runs=ydnpd.config.NUM_RUNS)

    dataset, schema = load_dataset(dataset_name)

    return task, task.execute(dataset, schema)
