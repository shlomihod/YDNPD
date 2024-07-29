import itertools as it
from math import comb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


EVALUATION_METRICS = [
    "marginals_3_max_abs_diff_error",
    "marginals_3_avg_abs_diff_error",
    "thresholded_marginals_3_max_abs_diff_error",
    "thresholded_marginals_3_avg_abs_diff_error",
    "corr_max_abs_diff",
    "corr_avg_abs_diff",
    "accuracy_diff",
    "auc_diff",
]

RANDOM_SEED = 42


def calc_k_marginals_abs_diff_errors(
    first_dataset: pd.DataFrame, second_dataset: pd.DataFrame, marginals_k: int
) -> dict[str, int]:

    columns = list(first_dataset.columns)

    marginals_abs_diff_errors = []

    def count_fn(dataset: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return dataset.groupby(keys).size().to_frame("count").reset_index()

    for keys in it.combinations(columns, marginals_k):
        keys = list(keys)

        marginals_abs_diff_errors.extend(
            pd.merge(
                count_fn(first_dataset, keys),
                count_fn(second_dataset, keys),
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
        / len(first_dataset),
        f"marginals_{marginals_k}_avg_abs_diff_error": np.sum(marginals_abs_diff_errors)
        / (len(first_dataset) * query_count),
    }


def calc_thresholded_marginals_k_abs_diff_errors(
    first_dataset: pd.DataFrame,
    second_dataset: pd.DataFrame,
    schema: dict,
    marginals_k: int,
) -> dict[str, int]:
    datasets = [first_dataset.copy(), second_dataset.copy()]

    for column in first_dataset.columns:
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
    first_dataset: pd.DataFrame,
    second_dataset: pd.DataFrame,
    split_proportion: float,
    target_column: str,
) -> dict[str, float]:

    split_mask = np.random.default_rng(seed=RANDOM_SEED).choice(
        [True, False],
        len(first_dataset),
        p=[split_proportion, 1 - split_proportion],
    )

    first_dataset_train = first_dataset[split_mask]
    first_dataset_test = first_dataset[~split_mask]

    X_train_first, y_train_first = (
        first_dataset_train.drop(columns=[target_column]),
        first_dataset_train[target_column],
    )
    X_test_first, y_test_first = (
        first_dataset_test.drop(columns=[target_column]),
        first_dataset_test[target_column],
    )
    X_train_second, y_train_second = (
        second_dataset.drop(columns=[target_column]),
        second_dataset[target_column],
    )

    model_first = RandomForestClassifier().fit(X_train_first, y_train_first)

    model_second = RandomForestClassifier().fit(X_train_second, y_train_second)

    accuracy_train_dataset = model_first.score(X_test_first, y_test_first)
    accuracy_other_dataset = model_second.score(X_test_first, y_test_first)

    # Function to prepare predictions for ROC AUC calculation
    def prepare_predictions(y_pred):
        unique_classes = np.unique(y_pred)
        if len(unique_classes) > 2:
            # One-hot encode the predictions
            encoder = OneHotEncoder(sparse=False)
            y_pred_one_hot = encoder.fit_transform(y_pred.reshape(-1, 1))
            return y_pred_one_hot
        else:
            # For binary classification, ensure predictions are in the right shape
            return y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred

    y_pred_first = model_first.predict(X_test_first)
    y_pred_second = model_second.predict(X_test_first)

    # Prepare predictions
    y_pred_first_prepared = prepare_predictions(y_pred_first)
    y_pred_second_prepared = prepare_predictions(y_pred_second)

    # Calculate AUC scores
    y_test_first_np = y_test_first.to_numpy()
    if len(np.unique(y_test_first_np)) > 2:
        auc_train_dataset = roc_auc_score(
            y_test_first_np, y_pred_first_prepared, multi_class="ovo"
        )
        auc_other_dataset = roc_auc_score(
            y_test_first_np, y_pred_second_prepared, multi_class="ovo"
        )
    else:
        auc_train_dataset = roc_auc_score(y_test_first_np, y_pred_first_prepared)
        auc_other_dataset = roc_auc_score(y_test_first_np, y_pred_second_prepared)

    auc_diff = auc_train_dataset - auc_other_dataset

    return {
        "accuracy_train_dataset": accuracy_train_dataset,
        "accuracy_other_dataset": accuracy_other_dataset,
        "accuracy_diff": accuracy_train_dataset - accuracy_other_dataset,
        "auc_train_dataset": auc_train_dataset,
        "auc_other_dataset": auc_other_dataset,
        "auc_diff": auc_diff,
    }


def calculate_corr(first_dataset: pd.DataFrame, second_dataset: pd.DataFrame) -> dict:
    first_dataset_corr = np.tril(first_dataset.corr(), k=-1)
    second_dataset_corr = np.tril(second_dataset.corr(), k=-1)

    abs_diff_corr = np.abs(first_dataset_corr - second_dataset_corr)
    num_corrs = comb(len(first_dataset.columns), 2)

    return {
        "corr_max_abs_diff": np.max(abs_diff_corr),
        "corr_avg_abs_diff": np.sum(abs_diff_corr) / num_corrs,
    }


def evaluate_two(
    first_dataset: pd.DataFrame,
    second_dataset: pd.DataFrame,
    schema: dict,
    classification_target_column: str,
    classification_split_proportion: float,
    marginals_k: int,
) -> dict:

    assert first_dataset.shape == second_dataset.shape
    assert list(first_dataset.columns) == list(second_dataset.columns)

    return (
        calc_k_marginals_abs_diff_errors(first_dataset, second_dataset, marginals_k)
        | calc_thresholded_marginals_k_abs_diff_errors(
            first_dataset, second_dataset, schema, marginals_k
        )
        | calculate_corr(first_dataset, second_dataset)
        | calc_classification_accuracy(
            first_dataset,
            second_dataset,
            classification_split_proportion,
            classification_target_column,
        )
    )
