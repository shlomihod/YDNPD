import json
from pathlib import Path

import pandas as pd
import numpy as np

from ydnpd.dataset import load_dataset, split_train_eval_datasets


RADNOM_SEED_GENERATION = 42


def save_dataset(dataset, schema, data_path, name):
    data_path = Path(data_path)
    dataset_path = data_path / name
    print(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path / f"{name}.csv", index=False)
    json.dump(schema, open(dataset_path / f"{name}.json", "w"), indent=2)


def generate_baseline_domain(dataset, schema, null_prop=None):

    rng = np.random.default_rng(RADNOM_SEED_GENERATION)

    schema = schema["schema"]
    num_records = len(dataset)

    generated_dataset = {}

    for column in dataset.columns:

        if column not in schema:
            raise ValueError(f"Column '{column}' not found in schema")

        if "values" not in schema[column]:
            raise ValueError(f"Column '{column}' must have 'values' key")

        values = [v for v in schema[column]["values"]]

        if schema[column].get("has_null", False):
            null_value = schema[column]["null_value"]
            values.remove(null_value)

        sampled_values = rng.choice(values, num_records)

        if schema[column].get("has_null", False) and null_prop is not None:
            mask = rng.random(num_records) < null_prop
            sampled_values[mask] = schema[column]["null_value"]

        generated_dataset[column] = sampled_values

    return pd.DataFrame(generated_dataset)


def generate_baseline_univariate(dataset, schema, rounding):

    train_dataet, _ = split_train_eval_datasets(dataset)

    if rounding is None:
        df = pd.DataFrame(
            {
                column: (
                    values.sample(
                        n=len(dataset), replace=True, random_state=RADNOM_SEED_GENERATION
                    ).reset_index(drop=True)
                )
                for column, values in train_dataet.items()
            }
        )

    else:

        rng = np.random.default_rng(RADNOM_SEED_GENERATION)

        pseudo_probs = {column:
                        dataset[column]
                        .value_counts(normalize=True)
                        .round(rounding)
                        for column in dataset.columns}
        probs = {column: (pp / pp.sum()).to_dict() for column, pp in pseudo_probs.items()}
        df = pd.DataFrame(
            {
                column: (
                    rng.choice(
                        list(ps.keys()),
                        size=len(dataset),
                        p=list(ps.values()))
                )
                for column, ps in probs.items()
            }
        )

    return df


def create_baselines(dataset_name, data_path, null_prop=None, rounding=2):
    dataset, schema = load_dataset(dataset_name, drop_na=null_prop is None)

    baselines = {
        "baseline_domain": generate_baseline_domain(
            dataset, schema, null_prop=null_prop
        ),
        "baseline_univariate": generate_baseline_univariate(dataset, schema, rounding),
    }

    for name, dataset in baselines.items():
        save_dataset(dataset, schema, data_path, name)


def create_upsamped(dataset_name, other_dataset_name, data_path):
    other_dataset, schema = load_dataset(other_dataset_name)
    num_records = len(other_dataset)

    dataset, schema = load_dataset(dataset_name)

    upsampled_dataset = dataset.sample(
        num_records, replace=True, random_state=RADNOM_SEED_GENERATION
    )

    family, dataset_core_name = dataset_name.split("/")
    name = f"{dataset_core_name}_upsampled"

    save_dataset(upsampled_dataset, schema, data_path, name)


if __name__ == "__main__":
    create_baselines("acs/national", "data/acs-2019-nist")
    create_upsamped("acs/massachusetts", "acs/national", "data/acs-2019-nist")
    create_upsamped("acs/texas", "acs/national", "data/acs-2019-nist")
