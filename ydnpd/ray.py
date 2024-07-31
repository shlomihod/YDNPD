import traceback
from functools import partial

import ray

from ydnpd import evaluate_two, UtilityTask
import ydnpd.config


def span_hparam_tasks(task_kwargs=None):

    if task_kwargs is None:
        task_kwargs = {}

    evaluation_fn = partial(
        evaluate_two,
        classification_target_column=ydnpd.config.CLASSIFICATION_TARGET_COLUMN,
        classification_split_proportion=ydnpd.config.CLASSIFICATION_SPLIT_PROPORTION,
        marginals_k=ydnpd.config.MARGINALS_K,
    )

    return [
        UtilityTask(
            dataset_name=dataset_name,
            epsilons=ydnpd.config.EPSILONS,
            synth_name=synth_name,
            hparam_dims=ydnpd.config.HPARAMS_DIMS[synth_name],
            evaluation_fn=evaluation_fn,
            num_runs=ydnpd.config.NUM_RUNS,
            **task_kwargs
        )
        for synth_name in ydnpd.config.EXPERIMENT_SYNTHESIZERS
        for dataset_name in ydnpd.config.DATASET_NAMES
    ]


def span_hparam_ray_tasks(**task_kwargs):

    def task_execute_wrapper(task):
        def function():
            return task.execute()

        return function

    return [
        ray.remote(task_execute_wrapper(task))
        .options(num_gpus=(1 if task.synth_name in ("patectgan") else 0))
        .remote()
        for task in span_hparam_tasks(task_kwargs)
    ]
