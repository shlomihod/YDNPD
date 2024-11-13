import sys
import traceback
from multiprocessing import Process, Queue
from typing import Any, Optional
import time

import torch
import pyro
import pyro.distributions as dist
import networkx as nx
import pandas as pd
import pandera as pa


MAX_TIMEOUT_SAMPLING = 10
MAX_SAMPLING_CHECKS = 10


def extract_single(answer):
    if len(answer) != 1:
        raise ValueError()
    return answer[0]


def clean_split(s, sep=","):
    return [item.strip() for item in s.split(sep)]


def build_graph(relationships):
    G = nx.DiGraph()
    for relationship in relationships:
        source, target = map(str.strip, relationship.split("->"))
        G.add_edge(source, target)

    return G


def metadata_to_pandera_schema(metadata_schema: dict[str, Any]) -> pa.DataFrameSchema:
    schema_dict = {}

    for column_name, column_info in metadata_schema.items():
        dtype = column_info["dtype"]
        values = column_info.get("values")
        checks = []

        if isinstance(values, list):
            checks.append(pa.Check.isin(values))
        elif isinstance(values, dict):
            allowed_values = list(values.keys())
            checks.append(pa.Check.isin(allowed_values))

        schema_dict[column_name] = pa.Column(
            dtype=dtype,
            checks=checks,
            title=column_info.get("description", column_name),
            nullable=False
        )

    schema = pa.DataFrameSchema(
        schema_dict,
        strict=True
    )

    return schema


def sample_dataset(model, num_samples, pandera_schema):
    records = []
    for _ in range(num_samples):
        sample = model()
        if sample is None:
            raise ValueError("Model sampled a None value")
        record = {key: value if isinstance(value, (int, float)) else value.item()
                  for key, value in sample.items()}
        records.append(record)

    df = pd.DataFrame(records)

    if pandera_schema is not None:
        pandera_schema.validate(df)

    return df


def retrieve_pyro_model(pyro_code):
    local_dict = {}

    local_dict.update({
        'pyro': pyro,
        'dist': dist,
        'torch': torch
    })

    exec(pyro_code, globals(), local_dict)
    model = local_dict['model']
    pyro.clear_param_store()
    # model_trace = pyro.poutine.trace(model).get_trace()

    return model


def _run_pyro_model_worker(queue: Queue, pyro_code: str, pandera_schema) -> None:
    """Worker function that creates and runs Pyro model in the subprocess."""
    try:
        model = retrieve_pyro_model(pyro_code)
        result = sample_dataset(model, MAX_SAMPLING_CHECKS, pandera_schema)
        for _ in range(MAX_SAMPLING_CHECKS):
            result = model()
        queue.put(("success", result))
    except Exception:
        queue.put(("error", traceback.format_exc(limit=1)))


def run_pyro_model_with_timeout(
    pyro_code: str,
    timeout: float,
    pandera_schema,
) -> tuple[bool, Any]:
    """
    Run Pyro model in a separate process with timeout.

    Args:
        pyro_code: The Pyro code to run
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, result_or_error: Any)
    """
    queue = Queue()
    process = Process(
        target=_run_pyro_model_worker,
        args=(queue, pyro_code, pandera_schema)
    )

    try:
        process.start()
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            if not process.is_alive():
                break
            time.sleep(0.1)

        if not queue.empty():
            status, result = queue.get_nowait()
            if status == "error":
                return False, result
            elif result is None:
                return False, "Model returned None instead of a sample"
            return True, result

        raise TimeoutError(f"Model execution timed out after {timeout} seconds")

    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
        process.close()


def is_valid_pyro_code(
    pyro_code: str, 
    pandera_schema: Optional[dict] = None,
    max_attempts: int = MAX_SAMPLING_CHECKS,
    sampling_timeout: int = MAX_TIMEOUT_SAMPLING,
) -> tuple[bool, Optional[str]]:
    """
    Validate Pyro code with process-based timeout handling.
    """
    try:
        # Verify model can be created before attempting runs
        retrieve_pyro_model(pyro_code)

        for _ in range(max_attempts):
            success, result = run_pyro_model_with_timeout(pyro_code, sampling_timeout, pandera_schema)
            if not success:
                return False, f"ERROR: {result}"

    except KeyboardInterrupt:
        raise
    except Exception:
        return False, traceback.format_exc(limit=1)
    else:
        return True, None
