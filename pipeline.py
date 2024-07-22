import os
import json
import datetime
import warnings
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sdnist.load
import sdnist.report

try: 
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
except ImportError:
    warnings.warn("SynthCity is not installed. Please install it to use SynthCity models.")

try:
    from sdv import single_table as sdv_single_table
    from sdv.metadata import SingleTableMetadata
except ImportError:
    warnings.warn("SDV is not installed. Please install it to use SDV models.")

try:
    from snsynth import Synthesizer as snsynth_Synthesizer
except ImportError:
    warnings.warn("SmartNoise is not installed. Please install it to use SmartNoise models.")


# https://pages.nist.gov/privacy_collaborative_research_cycle/pages/participate.html
COL_SUBSETS = {
    "demographic": ["SEX", "MSP", "RAC1P", "OWN_RENT", "PINCP_DECILE", "EDU"]  #, "AGEP", "HOUSING_TYPE", "DVET", "DEYE"]
}

BASELINE_DATASETS = ["baseline_domain", "baseline_univariate"]

sdnist.load.TestDatasetName = Enum("TestDatasetName", {x.name: x.value for x in sdnist.load.TestDatasetName}
                                   | {name: value
                                      for value, name in enumerate(BASELINE_DATASETS,
                                                                   start=len(sdnist.load.TestDatasetName)+1)})

sdnist.load.dataset_name_state_map |= {name: name for name in BASELINE_DATASETS}


DATASET_IDS = {"ma2019": "MA", "tx2019": "TX", "national2019": "NATIONAL",
               "baseline_domain": "BASELINE_DOMAIN", "baseline_univariate": "BASELINE_UNIVARIATE"}

DATASET_ENUMS = {"MA": sdnist.load.TestDatasetName.ma2019,
                 "TX": sdnist.load.TestDatasetName.tx2019,
                 "NATIONAL": sdnist.load.TestDatasetName.national2019,
                 "BASELINE_DOMAIN": sdnist.load.TestDatasetName.baseline_domain,
                 "BASELINE_UNIVARIATE": sdnist.load.TestDatasetName.baseline_univariate}


def load_dataest(dataset_name, cols_subset_name=None, root=sdnist.load.DEFAULT_DATASET, download=False):

    real_df, schema = sdnist.load.load_dataset(challenge="census",
                root=root,
                download=download,
                public=False,
                test=getattr(sdnist.load.TestDatasetName, dataset_name),
                format_="csv")

    if cols_subset_name is not None:
        col_subset = COL_SUBSETS[cols_subset_name]
        real_df = real_df[col_subset]
        schema = {"schema": {k: v for k, v in schema["schema"].items() if k in col_subset}}

    return real_df, schema


def generate_dataframe_uniformly_from(schema, num_records, cols_subset_name=None, keep_na_symbol=True, na_prop=0.05, random_seed=None):

    if "schema" in schema:
        schema = schema["schema"]

    prng = np.random.default_rng(random_seed)

    def sample_value(column_schema):
        if 'values' in column_schema:
            values = column_schema['values']
            if column_schema.get('has_null', False):
                values = [x for x in values if x != column_schema['null_value']]
            value = prng.choice(values)
        elif 'min' in column_schema and 'max' in column_schema:
            value = prng.integers(column_schema['min'], column_schema['max'] + 1)
        else:
            raise ValueError("Unsupported schema")

        # Handle null values if applicable
        if column_schema.get('has_null', False) and prng.random() < na_prop:   # % chance of null
            value = column_schema['null_value'] if keep_na_symbol else pd.NA
        return value

    data = {}
    for column, column_schema in schema.items():
        if cols_subset_name is None or column in COL_SUBSETS[cols_subset_name]:
            data[column] = [sample_value(column_schema) for _ in range(num_records)]

    # Convert to DataFrame and ensure the correct dtypes
    df = pd.DataFrame(data)
    for column, column_schema in schema.items():
        if column_schema['dtype'] == 'int64':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif column_schema['dtype'] == 'int32':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int32')
        else:
            df[column] = df[column].astype(column_schema['dtype'])

    return df


def generate_dataframe_univarietly_from(df):
    return pd.DataFrame({col: df[col].sample(n=len(df), replace=True).values for col in df.columns})


def create_baseline_dataset(path, dataset_name, col_subset_name=None):
    path = Path(path)
    df, schema = load_dataest(dataset_name, col_subset_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path / "baseline_domain", exist_ok=True)
    os.makedirs(path / "baseline_univariate", exist_ok=True)
    generate_dataframe_uniformly_from(schema, len(df)).to_csv(path / "baseline_domain" / "baseline_domain.csv", index=False)
    generate_dataframe_univarietly_from(df).to_csv(path / "baseline_univariate" / "baseline_univariate.csv", index=False)


def generate_synth_data(real_df, schema, model_name, epsilon=None):

    num_samples = len(real_df)

    if model_name == "id":
        synth_df = real_df.copy()

    # SDV
    elif model_name in ["CTGAN", "CopulaGAN", "GaussianCopula", "TVAE"]:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_df)

        model = getattr(sdv_single_table, f"{model_name}Synthesizer")(metadata)
        model.fit(real_df)
        synth_df = model.sample(num_samples)

    # SmartNoise
    elif model_name in ["aim", "mwem", "mst", "pacsynth", "dpctgan", "patectgan"]:
        continuous_columns = []
        categorical_columns = []

        for col, info in schema["schema"].items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case ("int32" | "int64"):
                    if len(real_df[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        model = snsynth_Synthesizer.create(model_name, epsilon=epsilon, verbose=True)
        model.fit(real_df, preprocessor_eps=0.5,
                    continuous_columns=continuous_columns,
                    categorical_columns=categorical_columns)

        synth_df = model.sample(num_samples)

    # Synthcity
    elif model_name in ['dpgan', 'privbayes', 'adsgan', 'decaf', 'pategan', 'AIM']:
        loader = GenericDataLoader(
            real_df
        )

        model = Plugins().get(model_name.lower(), epsilon=epsilon)

        model.fit(loader)

        synth_df = model.generate(count=num_samples).dataframe()

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return synth_df


@contextmanager
def no_pylot_display():
    # Save the current backend
    original_backend = plt.get_backend()

    try:
        # Switch to the 'Agg' backend to prevent plots from displaying
        plt.switch_backend('Agg')
        yield
    finally:
        # Switch back to the original backend
        plt.switch_backend(original_backend)


def run_sdnist_evaluation(synth_df, dataset_name, experiment_id, show_report=False):
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    synthetic_filepath = results_path / (f"{experiment_id}.csv")
    synth_df.to_csv(synthetic_filepath, index=False)

    dataset_enum = DATASET_ENUMS[DATASET_IDS[dataset_name]]
    download = False
    labels_dict = None
    data_root = Path(sdnist.load.DEFAULT_DATASET)

    time_now = datetime.datetime.now().strftime('%m-%d-%YT%H.%M.%S')
    this_report_dir = Path(sdnist.report.REPORTS_DIR, f'{dataset_enum.name}_{time_now}')
    output_directory = Path(this_report_dir)
    ui_data = sdnist.report.ReportUIData(output_directory=output_directory)
    report_data = sdnist.report.ReportData(output_directory=output_directory)

    outfile = Path(output_directory, 'report.json')

    if not sdnist.report.REPORTS_DIR.exists():
        os.mkdir(sdnist.report.REPORTS_DIR)
    if not this_report_dir.exists():
        os.mkdir(this_report_dir)

    log = sdnist.utils.SimpleLogger()
    log.msg('SDNist: Deidentified Data Report Tool', level=0, timed=False)
    log.msg(f'Creating Evaluation Report for Deidentified Data at path: {synthetic_filepath}',
            level=1)

    if not outfile.exists():
        log.msg('Loading Datasets', level=2)
        dataset = sdnist.report.Dataset(synthetic_filepath, log, dataset_enum, data_root, download)

        ui_data = sdnist.report.dataset.data_description(dataset, ui_data, report_data, labels_dict)
        log.end_msg()

        # Create scores
        log.msg('Computing Utility Scores', level=2)
        with no_pylot_display():
            ui_data, report_data = sdnist.report.utility_score(dataset, ui_data, report_data, log)
        log.end_msg()

        log.msg('Computing Privacy Scores', level=2)
        ui_data, report_data = sdnist.report.privacy_score(dataset, ui_data, report_data, log)
        log.end_msg()

        log.msg('Saving Report Data')
        ui_data.save()
        ui_data = ui_data.data
        report_data.data['created_on'] = ui_data['Created on']
        report_data.save()
        log.end_msg()
    else:
        with open(outfile, 'r') as f:
            ui_data = json.load(f)
        log.end_msg()
        # Generate Report

    sdnist.report.generate(ui_data, output_directory, show_report)
    log.msg(f'Reports available at path: {output_directory}', level=0, timed=False,
            msg_type='important')

    return report_data


def load_generate_evaluate(dataset_name, cols_subset_name, model_name, epsilon=None, show_report=False):
    expermint_id = {dataset_name}-{cols_subset_name}-{model_name}-{str(epsilon).replace('.', '_')}
    real_df, schema = load_dataest(dataset_name, cols_subset_name)
    synth_df = generate_synth_data(real_df, schema, model_name, epsilon=epsilon)
    evaluation = run_sdnist_evaluation(synth_df, dataset_name, expermint_id, show_report=show_report)
    return evaluation
