import itertools as it

import numpy as np
import pandas as pd


def evaluate(first_dataset: pd.DataFrame, second_dataset: pd.DataFrame,
             marginals_up_to_k=3) -> dict:

    assert first_dataset.shape == second_dataset.shape
    assert list(first_dataset.columns) == list(second_dataset.columns)

    columns = list(first_dataset.columns)

    marginals_abs_diff_errors = []

    def count_fn(dataset: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return dataset.groupby(keys).size().to_frame("count").reset_index()

    for keys in it.combinations(columns, marginals_up_to_k):
        keys = list(keys)
        marginals_abs_diff_errors.extend(
            pd.merge(count_fn(first_dataset, keys),
                     count_fn(second_dataset, keys),
                     how="outer",
                     on=keys)
            .fillna(0)
            .apply(lambda row: np.abs(row["count_x"] - row["count_y"]), axis=1)
            .to_list()
        )

    return {
        "marginals_up_to_3_max_abs_diff_errors": np.max(marginals_abs_diff_errors)
    }
