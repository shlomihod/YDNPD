from functools import partial

import ray

import ydnpd


def span_utility_tasks(task_kwargs=None):

    if task_kwargs is None:
        task_kwargs = {}

    return [
        ydnpd.harness.UtilityTask(
            dataset_name=dataset_name,
            epsilons=ydnpd.harness.config.EPSILONS,
            synth_name=synth_name,
            hparam_dims=ydnpd.harness.config.HPARAMS_DIMS[synth_name],
            num_runs=ydnpd.harness.config.NUM_RUNS,
            evaluation_kwargs=ydnpd.harness.config.EVALUATION_KWARGS,
            **task_kwargs
        )
        for synth_name in ydnpd.harness.config.SYNTHESIZERS
        for dataset_name in ydnpd.harness.config.DATASET_NAMES
    ]


def span_utility_ray_tasks(**task_kwargs):

    def task_execute_wrapper(task):
        def function():
            return task.execute()

        return function

    return [
        ray.remote(task_execute_wrapper(task))
        .options(num_gpus=(1 if task.synth_name in ("patectgan", "aim_torch") else 0))
        .remote()
        for task in span_utility_tasks(task_kwargs)
    ]
