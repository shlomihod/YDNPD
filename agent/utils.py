import traceback
from multiprocessing import Process, Queue
from typing import Any, Optional, Tuple
import time

import torch
import pyro
import pyro.distributions as dist
import networkx as nx


MAX_TIMEOUT_SAMPLING = 5


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


def _run_pyro_model_worker(queue: Queue, pyro_code: str) -> None:
    """Worker function that creates and runs Pyro model in the subprocess."""
    try:
        model = retrieve_pyro_model(pyro_code)
        result = model()
        queue.put(("success", result))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def run_pyro_model_with_timeout(
    pyro_code: str,
    timeout: float,
) -> Tuple[bool, Any]:
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
        args=(queue, pyro_code)
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
    *, 
    max_attempts: int = 10,
    sample_timeout: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validate Pyro code with process-based timeout handling.
    """
    try:
        # Verify model can be created before attempting runs
        retrieve_pyro_model(pyro_code)

        for attempt in range(max_attempts):
            success, result = run_pyro_model_with_timeout(pyro_code, sample_timeout)
            if not success:
                return False, f"ERROR: {result}"

    except KeyboardInterrupt:
        raise
    except Exception:
        return False, traceback.format_exc()

    return True, None
