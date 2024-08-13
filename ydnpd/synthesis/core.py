import warnings

import pandas as pd

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
except ImportError:
    warnings.warn(
        "SynthCity is not installed. Please install it to use SynthCity synthesizers."
    )

try:
    from sdv import single_table as sdv_single_table
    from sdv.metadata import SingleTableMetadata
except ImportError:
    warnings.warn("SDV is not installed. Please install it to use SDV synthesizers.")

try:
    from snsynth import Synthesizer as snsynth_Synthesizer
except ImportError:
    warnings.warn(
        "SmartNoise is not installed. Please install it to use SmartNoise synthesizers."
    )

from ydnpd.synthesis.privbayes import PrivBayes
from ydnpd.synthesis.aim_torch import AIMSynthesizerTorch

SYNTHESIZERS = [
    "id",
    "CTGAN",
    "CopulaGAN",
    "GaussianCopula",
    "TVAE",
    "aim",
    "aim_torch",
    "mwem",
    "mst",
    "pacsynth",
    "dpctgan",
    "patectgan",
    "dpgan",
    "privbayes",
    "adsgan",
    "decaf",
    "pategan",
    "AIM",
]


def generate_synthetic_data(
    dataset: pd.DataFrame, schema: dict, epsilon: float, synth_name: str, **hparams
):

    num_samples = len(dataset)

    if synth_name == "id":
        synth_df = dataset.copy()

    # SDV
    elif synth_name in ["CTGAN", "CopulaGAN", "GaussianCopula", "TVAE"]:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset)

        synthesizer = getattr(sdv_single_table, f"{synth_name}Synthesizer")(
            metadata, **hparams
        )
        synthesizer.fit(dataset)
        synth_df = synthesizer.sample(num_samples)

    # SmartNoise
    elif synth_name in ["aim", "mwem", "mst", "pacsynth", "dpctgan", "patectgan"]:
        continuous_columns = []
        categorical_columns = []

        for col, info in schema["schema"].items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case "int32" | "int64":
                    if len(dataset[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        preprocessor_eps = hparams.pop("preprocessor_eps")
        synthesizer = snsynth_Synthesizer.create(
            synth_name, epsilon=epsilon, verbose=True, **hparams
        )
        synthesizer.fit(
            dataset,
            preprocessor_eps=preprocessor_eps,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )

        synth_df = synthesizer.sample(num_samples)

    # Synthcity
    elif synth_name in ["dpgan", "adsgan", "decaf", "pategan", "AIM"]:
        loader = GenericDataLoader(synth_name)

        synthesizer = Plugins().get(synth_name.lower(), epsilon=epsilon, **hparams)

        synthesizer.fit(loader)

        synth_df = synthesizer.generate(count=num_samples).dataframe()

    elif synth_name == "privbayes":
        synthesizer = PrivBayes(epsilon=epsilon, **hparams)
        synth_df = synthesizer.fit_sample(dataset, schema)

    elif synth_name == "aim_torch":
        continuous_columns = []
        categorical_columns = []

        for col, info in schema["schema"].items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case "int32" | "int64":
                    if len(dataset[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        preprocessor_eps = hparams.pop("preprocessor_eps")
        synthesizer = AIMSynthesizerTorch(epsilon=epsilon, verbose=True, **hparams)
        synthesizer.fit(
            dataset,
            preprocessor_eps=preprocessor_eps,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        synth_df = synthesizer.sample(num_samples)

    else:
        raise ValueError(f"Unknown synthesizer name: {synth_name}")

    return synth_df
