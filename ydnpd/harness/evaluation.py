import itertools as it
from math import comb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats.contingency import association


EVALUATION_METRICS = [
    "marginals_3_max_abs_diff_error",
    "marginals_3_avg_abs_diff_error",
    "thresholded_marginals_3_max_abs_diff_error",
    "thresholded_marginals_3_avg_abs_diff_error",
    "pearson_corr_max_abs_diff",
    "pearson_corr_avg_abs_diff",
    "cramer_v_corr_max_abs_diff",
    "cramer_v_corr_avg_abs_diff",
    "accuracy_diff",
    "auc_diff",
]

RANDOM_SEED = 42


def _cramers_v_matrix(df):
    cols = df.columns
    n = len(cols)
    cramers_v_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cramers_v_mat[i, j] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[cols[i]], df[cols[j]])
                cramers_v_mat[i, j] = association(confusion_matrix, method="cramer")
    return pd.DataFrame(cramers_v_mat, index=cols, columns=cols)


def calc_k_marginals_abs_diff_errors(
    train_dataset: pd.DataFrame, synth_dataset: pd.DataFrame, marginals_k: int
) -> dict[str, int]:

    columns = list(train_dataset.columns)

    marginals_abs_diff_errors = []

    def count_fn(dataset: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return dataset.groupby(keys).size().to_frame("count").reset_index()

    for keys in it.combinations(columns, marginals_k):
        keys = list(keys)

        marginals_abs_diff_errors.extend(
            pd.merge(
                count_fn(train_dataset, keys),
                count_fn(synth_dataset, keys),
                how="outer",
                on=keys,
            )
            .fillna(0)
            .apply(lambda row: np.abs(row["count_x"] - row["count_y"]), axis=1)
            .to_list()
        )

    query_count = comb(len(columns), marginals_k)

    return {
        f"marginals_{marginals_k}_max_abs_diff_error": np.max(marginals_abs_diff_errors)
        / len(train_dataset),
        f"marginals_{marginals_k}_avg_abs_diff_error": np.sum(marginals_abs_diff_errors)
        / (len(train_dataset) * query_count),
    }


def calc_thresholded_marginals_k_abs_diff_errors(
    train_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    schema: dict,
    marginals_k: int,
) -> dict[str, int]:
    datasets = [train_dataset.copy(), synth_dataset.copy()]

    for column in train_dataset.columns:
        column_schema = schema["schema"][column]
        if (values := column_schema.get("values")) is not None:
            assert column_schema["dtype"].startswith(
                "int"
            ), "Only integer columns can be categorical; you might have missing values"

            assert all(
                isinstance(v, int) for v in values
            ), "Categorical values must be integers"
            assert len(values) == len(set(values)), "Categorical values must be unique"
            assert values == sorted(values), "Categorical values must be sorted"

            mid_value = values[len(values) // 2]

            for dataset in datasets:

                # if the value is less than the mid value, set it to 0, otherwise set it to 1
                dataset[column] = (dataset[column] < mid_value).astype(int)

        args = datasets + [marginals_k]

    return {
        f"thresholded_{k}": v
        for k, v in calc_k_marginals_abs_diff_errors(*args).items()
    }


def calc_classification_accuracy(
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:

    X_train, y_train = (
        train_dataset.drop(columns=[target_column]),
        train_dataset[target_column],
    )
    X_eval, y_eval = (
        eval_dataset.drop(columns=[target_column]),
        eval_dataset[target_column],
    )
    X_synth, y_synth = (
        synth_dataset.drop(columns=[target_column]),
        synth_dataset[target_column],
    )

    model_train = RandomForestClassifier().fit(X_train, y_train)
    model_synth = RandomForestClassifier().fit(X_synth, y_synth)

    accuracy_train_dataset = model_train.score(X_eval, y_eval)
    accuracy_other_dataset = model_synth.score(X_eval, y_eval)

    # Use predict_proba instead of predict to get probabilities
    y_pred_proba_train = model_train.predict_proba(X_eval)
    y_pred_proba_synth = model_synth.predict_proba(X_eval)

    # Calculate AUC scores
    y_eval_np = y_eval.to_numpy()
    if len(np.unique(y_eval_np)) > 2:
        auc_train_dataset = roc_auc_score(
            y_eval_np, y_pred_proba_train, multi_class="ovo"
        )
        auc_synth_dataset = roc_auc_score(
            y_eval_np, y_pred_proba_synth, multi_class="ovo"
        )
    else:
        auc_train_dataset = roc_auc_score(y_eval_np, y_pred_proba_train[:, 1])
        auc_synth_dataset = roc_auc_score(y_eval_np, y_pred_proba_synth[:, 1])

    auc_diff = auc_train_dataset - auc_synth_dataset

    return {
        "accuracy_train_dataset": accuracy_train_dataset,
        "accuracy_other_dataset": accuracy_other_dataset,
        "accuracy_diff": accuracy_train_dataset - auc_synth_dataset,
        "auc_train_dataset": auc_train_dataset,
        "auc_synth_dataset": auc_synth_dataset,
        "auc_diff": auc_diff,
    }


def calculate_corr(train_dataset: pd.DataFrame, synth_dataset: pd.DataFrame) -> dict:
    train_dataset_pearson_corr = np.tril(train_dataset.corr(), k=-1)
    synth_dataset_pearson_corr = np.tril(synth_dataset.corr(), k=-1)
    abs_diff_pearson_corr = np.abs(
        train_dataset_pearson_corr - synth_dataset_pearson_corr
    )

    train_dataset_cramer_v_corr = np.tril(_cramers_v_matrix(train_dataset), k=-1)
    synth_dataset_cramer_v_corr = np.tril(_cramers_v_matrix(synth_dataset), k=-1)
    abs_diff_cramer_v_corr = np.abs(
        train_dataset_cramer_v_corr - synth_dataset_cramer_v_corr
    )

    num_corrs = comb(len(train_dataset.columns), 2)

    return {
        "pearson_corr_max_abs_diff": np.max(abs_diff_pearson_corr),
        "pearson_corr_avg_abs_diff": np.sum(abs_diff_pearson_corr) / num_corrs,
        "cramer_v_corr_max_abs_diff": np.max(abs_diff_cramer_v_corr),
        "cramer_v_corr_avg_abs_diff": np.sum(abs_diff_cramer_v_corr) / num_corrs,
    }


def evaluate_two(
    train_dataset: pd.DataFrame,
    eval_dataset: pd.DataFrame,
    synth_dataset: pd.DataFrame,
    schema: dict,
    classification_target_column: str,
    marginals_k: int,
) -> dict:

    assert train_dataset.shape == synth_dataset.shape
    assert list(train_dataset.columns) == list(synth_dataset.columns)

    return (
        calc_k_marginals_abs_diff_errors(train_dataset, synth_dataset, marginals_k)
        | calc_thresholded_marginals_k_abs_diff_errors(
            train_dataset, synth_dataset, schema, marginals_k
        )
        | calculate_corr(train_dataset, synth_dataset)
        | calc_classification_accuracy(
            train_dataset,
            eval_dataset,
            synth_dataset,
            classification_target_column,
        )
    )
