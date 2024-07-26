

DATASET_NAMES = ["national", "massachusetts", "baseline_domain", "baseline_univariate"]

EPSILONS = [1, 4, 10]

NUM_RUNS = 5

FIXED_PREPROCESSOR_EPSILON = 10_000

EXPERIMENT_SYNTHESIZERS  = ["privbayes", "mwem", "aim"]

HPARAMS_DIMS = {
    "mwem": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "q_count": [128, 512],
        "marginal_width": [2, 3],
        "iterations": [10, 50],
        "add_ranges": [False], # True],
        "split_factor": [None], #, 3],
        "mult_weights_iterations": [20],
    },
    "mst": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
    },
    "aim": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "degree": [2, 3],
        "rounds": [10, 100],
    },
    "privbayes": {
        "theta": [2, 4, 8, 16, 32, 64],
        "epsilon_split": [0.1, 0.5, 0.75],
    },
    "patectgan": {
        "preprocessor_eps": [FIXED_PREPROCESSOR_EPSILON],
        "embedding_dim": [128],
        "generator_dim": [(256, 256)],
        "discriminator_dim": [(256, 256)],
        "epochs": [300],
        "generator_lr": [2e-4, 2e-5],
        "discriminator_lr": [2e-4, 2e-5],
        "generator_decay": [1e-6],
        "discriminator_decay": [1e-6],
        "batch_size": [500],
        "noise_multiplier": [1e-3, 0.1, 1, 5],
        "loss": ["cross_entropy", "wasserstein"],
        "teacher_iters": [5],
        "student_iters": [5],
        "sample_per_teacher": [1000],
        "delta": [None],
        "moments_order": [100],
        # discriminator_steps: NOT BEING USED
    },
}

#                dev                    test
EXPERIMENTS = [("massachusetts",       "national"),
               ("baseline_domain",     "national"),
               ("baseline_univariate", "national")]

CLASSIFICATION_TARGET_COLUMN = "OWN_RENT"
CLASSIFICATION_SPLIT_PROPORTION = 0.7
MARGINALS_UP_TO_K = 3
