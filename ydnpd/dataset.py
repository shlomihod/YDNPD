import json
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("diverse_communities_data_excerpts")

DATASETS = {"national": "national/national2019",
            "massachusetts": "massachusetts/ma2019",
            "texas": "texas/tx2019",
            "baseline_domain": "baseline_domain/baseline_domain",
            "baseline_univariate": "baseline_univariate/baseline_univariate",}

# https://pages.nist.gov/privacy_collaborative_research_cycle/pages/participate.html
COL_SUBSETS = {
    "demographic": ["SEX", "MSP", "RAC1P", "OWN_RENT", "PINCP_DECILE", "EDU"]  #, "AGEP", "HOUSING_TYPE", "DVET", "DEYE"]
}


def load_dataset(dataset_name: str, cols_subset_name: str = "demographic", drop_na: bool = True):
    dataset_path = DATASET_ROOT / DATASETS[dataset_name]
    dataset = pd.read_csv(dataset_path.with_suffix(".csv"))
    schema = json.load(open(dataset_path.with_suffix(".json")))

    if cols_subset_name is not None:
        col_subset = COL_SUBSETS[cols_subset_name]
        dataset = dataset[col_subset]
        schema = {"schema": {k: v for k, v in schema["schema"].items() if k in col_subset}}

    if drop_na:
        for col in dataset.columns:
            col_schema = schema["schema"][col]
            if col_schema.pop("has_null", False):
                null_value = col_schema.pop("null_value")
                dataset = dataset[dataset[col] != null_value]
                col_schema["values"] = [int(v) for v in col_schema["values"] if v != null_value]
                col_schema["dtype"] = "int64"

    for col in dataset.columns:
        dataset[col] = dataset[col].astype(schema["schema"][col]["dtype"])

    return dataset, schema
