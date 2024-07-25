import traceback
from functools import partial
from collections import defaultdict

import ray

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

    try:
        result = task.execute(dataset, schema)
    except Exception as e:
        print(f"Error in {synth_name} on {dataset_name}: {e}")
        traceback.print_stack()
        result = None

    return task, dataset_name, result


def span_hparam_tasks():
    evaluation_fn = partial(evaluate_two,
                            classification_target_column=ydnpd.config.CLASSIFICATION_TARGET_COLUMN,
                            classification_split_proportion=ydnpd.config.CLASSIFICATION_SPLIT_PROPORTION,
                            marginals_up_to_k=ydnpd.config.MARGINALS_UP_TO_K)

    return [HyperParamSearchTask(epsilons=ydnpd.config.EPSILONS,
                                 synth_name=synth_name,
                                 hparam_dims=ydnpd.config.HPARAMS_DIMS[synth_name],
                                 evaluation_fn=evaluation_fn,
                                 num_runs=ydnpd.config.NUM_RUNS)
            for synth_name in ydnpd.config.EXPERIMENT_SYNTHESIZERS
            for dataset_name in ydnpd.config.DATASET_NAMES]   


def span_hparam_ray_tasks():
    return [ray.remote(ydnpd.run_hparam_task)
            # .option(num_gpus=(1 if synth_name in ("patectgan") else 0))
            .remote(synth_name, dataset_name)
            for synth_name in ydnpd.config.EXPERIMENT_SYNTHESIZERS
            for dataset_name in ydnpd.config.DATASET_NAMES]


def collect_hparam_runs(flatten_tasks_results):
    synth_dataset_epsilon_results = defaultdict(lambda: defaultdict(list))
    task_results = {}

    for task, dataset_name, results in flatten_tasks_results:
        for epsilon in map(str, task.epsilons):
            if (result := results.get(epsilon)) is not None:
                synth_dataset_epsilon_results[dataset_name][epsilon].extend(result)
            else:
                print(f"Error in {task.synth_name} on {dataset_name} at epsilon {epsilon}")

        task_results[task.synth_name] = (task, synth_dataset_epsilon_results)

    return task_results
