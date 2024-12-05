import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE_TRAIN_TEST_SPLIT = 42
EVAL_SPLIT_PROPORTION = 0.3

DATASET_ROOT = Path(__file__).parent / "data"

DATASETS = {
    "acs/national": "acs-2019-nist/national/national2019",
    "acs/massachusetts": "acs-2019-nist/massachusetts/ma2019",
    "acs/massachusetts_upsampled": "acs-2019-nist/massachusetts_upsampled/massachusetts_upsampled",
    "acs/texas": "acs-2019-nist/texas/tx2019",
    "acs/texas_upsampled": "acs-2019-nist/texas_upsampled/texas_upsampled",
    "acs/baseline_domain": "acs-2019-nist/baseline_domain/baseline_domain",
    "acs/baseline_univariate": "acs-2019-nist/baseline_univariate/baseline_univariate",
    "acs/llm": "acs-2019-nist/llm/llm",
    "we/original": "we_survey/cleaned/cleaned",
}

# https://pages.nist.gov/privacy_collaborative_research_cycle/pages/participate.html
# maybe only run on the categorical columns>?
COL_SUBSETS = {
    "acs": {
        "demographic": [
            "SEX",
            "MSP",
            "RAC1P",
            "OWN_RENT",
            "PINCP_DECILE",
            "EDU",
            "HOUSING_TYPE",
            # "DVET",  # Many missing values
            # "DEYE",
            # "AGEP",  # Continuous
        ]
    }
}


def load_dataset(
    dataset_name: str, cols_subset_name: str | bool = True, drop_na: bool = True
):

    dataset_family, _ = dataset_name.split("/")

    dataset_path = DATASET_ROOT / DATASETS[dataset_name]
    dataset = pd.read_csv(dataset_path.with_suffix(".csv"))
    schema = json.load(open(dataset_path.with_suffix(".json")))

    if cols_subset_name:
        if dataset_family == "acs":
            cols_subset_name = "demographic"
        elif dataset_family == "we":
            raise ValueError("cols_subset_name must be False for we datasets")

        col_subset = COL_SUBSETS[dataset_family][cols_subset_name]
        dataset = dataset[col_subset]
        schema = {
            "schema": {k: v for k, v in schema["schema"].items() if k in col_subset}
        }

    if drop_na:
        for col in dataset.columns:
            col_schema = schema["schema"][col]
            if col_schema.pop("has_null", False):
                null_value = col_schema.pop("null_value")
                dataset = dataset[dataset[col] != null_value]
                col_schema["values"] = [
                    int(v) for v in col_schema["values"] if v != null_value
                ]
                col_schema["dtype"] = "int64"

    for col in dataset.columns:
        dataset[col] = dataset[col].astype(schema["schema"][col]["dtype"])

    return dataset, schema


def split_train_eval_datasets(dataset):

    return train_test_split(
        dataset,
        test_size=EVAL_SPLIT_PROPORTION,
        random_state=RANDOM_STATE_TRAIN_TEST_SPLIT,
    )
