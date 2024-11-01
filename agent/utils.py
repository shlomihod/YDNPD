import traceback

import torch
import pyro
import pyro.distributions as dist
import networkx as nx


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


def is_valid_pyro_code(pyro_code):
    try:
        model = retrieve_pyro_model(pyro_code)
        model()
    except Exception:
        return False, traceback.format_exc()
    else:
        return True, None
