import os
import multiprocessing

from ydnpd import ALL_EXPERIMENTS
from ydnpd.harness.config import EPSILONS


SWEEP_ID = "7i09oqiw"
NUM_GPUS = 0
NUM_CPUS = 2


def get_sweep_config(dataset_family):

    public_dataaset_pointers = [name for name in ALL_EXPERIMENTS[dataset_family].dev_names
                                if name != ALL_EXPERIMENTS[dataset_family].test_name]

    return {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "auc"},
        "parameters": {
            "pre_num_epochs": {"values": [1, 3 ,9]},
            "pre_batch_size": {"values": [4, 32, 128]},
            "pre_lr": {"values": [3e-4, 3e-5]},
            "dp_num_epochs": {"value": [20]},
            "dp_batch_size": {"values": [64, 128, 256]},
            "dp_lr": {"values": [3e-3, 3e-4]},
            "epsilon": {"values": EPSILONS},
            "private_data_pointer": {"value": ALL_EXPERIMENTS[DATASET_FAMILY].test_name},
            "public_data_pointer": {"values": public_dataaset_pointers},
        },
    }


def runner():
    import wandb
    from ydnpd.pretraining.trainer import TransformerTrainer, ModelConfig, PreTrainConfig
    from ydnpd.pretraining.utils import set_strict_reproducibility_by_config

    wandb.init(project="ydnpd-dp-ft")
    print(wandb.config)
    set_strict_reproducibility_by_config(wandb.config)
    results = TransformerTrainer.train_and_evaluate(
        config=ModelConfig(
            num_epochs=wandb.config.dp_num_epochs,
            batch_size=wandb.config.dp_batch_size,
            lr=wandb.config.dp_lr,
            epsilon=wandb.config.epsilon
        ),
        pretrain_config=PreTrainConfig(
            num_epochs=wandb.config.pre_num_epochs,
            batch_size=wandb.config.pre_batch_size,
            lr=wandb.config.pre_lr,
        ),
        public_data_pointer=wandb.config.public_data_pointer,
        private_data_pointer=wandb.config.private_data_pointer,
    )
    wandb.log(results)


def run_agent(cuda_device, runner_fn):
    if cuda_device < NUM_GPUS:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    import wandb
    wandb.agent(SWEEP_ID, function=runner_fn, project="ydnpd-dp-ft")


if __name__ == '__main__':
    processes = []
    for device in range(NUM_CPUS):
        p = multiprocessing.Process(target=run_agent, args=(device, runner))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
