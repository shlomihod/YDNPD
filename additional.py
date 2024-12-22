from pathlib import Path

ADDITIONAL_PATH = "llm_datasets"

ADDITIONAL_EXPERIMENTS = {
    experiment_name:
    [
        (f"{experiment_name}/{path.stem}", ADDITIONAL_PATH)
        for path in Path(f"{ADDITIONAL_PATH}/{experiment_name}").glob("*.csv")
        ]
        for experiment_name in ["acs"]
}
