import openai
import re

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine
import traceback

## Prompting

def openai_prompt(message_history, client, text_only=True):
    try:
        response = client.chat.completions.create(
            messages=message_history,
            model="gpt-4o-mini", # "gpt-3.5-turbo", # gpt-4o-mini'
            # try with 4o NON mini on small set to check if boost is substantial
            temperature = 0.7,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0

        )
        if text_only:
            return response.choices[0].message.content
        else:
            return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# extract the answer between <Answer>...</Answer> tags
def extract_answer_list(response_text):
    match = re.search(r"<Answer>(.*?)</Answer>", response_text, re.DOTALL)
    if match:
        return [item.strip() for item in match.group(1).split(",")]
    return []

def extract_answer_raw(response_text):
    match = re.search(r"<Answer>(.*?)</Answer>", response_text, re.DOTALL)
    if match:
        return match.group(1)
    return None
    
def general_prompt(instruction_func, message_history, client, args={}, extract_answer=extract_answer_list):
    full_prompt = instruction_func(**args)

    message_history.append({
        "role": "user",
        "content": full_prompt
    })

    response = openai_prompt(message_history, client)
    
    extracted_answer = extract_answer(response)
    
    # NOTE: this could be adjusted with more specific response message
    response_content = f'\n Extracted answer from response: {extracted_answer}\n\n'

    print(response)
    print(response_content)

    message_history.append({
        "role": "assistant",
        "content": response
    })

    message_history.append({
        "role": "assistant",
        "content": response_content
    })
    
    return message_history, extracted_answer

## Network code

def contains_cycles(relationships):
    G_temp = nx.DiGraph()
    for relationship in relationships:
        source, target = map(str.strip, relationship.split("->"))
        G_temp.add_edge(source, target)
    return not nx.is_directed_acyclic_graph(G_temp)

## Pyro code

def extract_pyro_code(response_text):
    match = re.search(r"<PyroCode>(.*?)</PyroCode>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No Pyro code found in the response text.")

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
    model_trace = pyro.poutine.trace(model).get_trace()

    return pyro_code

def robust_retrieve_pyro_model(pyro_code, prompt_fix_this_code, message_history, client, max_retries=5):
    attempt = 0
    while attempt < max_retries:
        try:
            # attempt to sample from the Pyro model
            pyro_code = retrieve_pyro_model(pyro_code)
            print(f'Succeeded, attempt {attempt}, final Pyro code:')
            print(pyro_code)
            return pyro_code, attempt
        except Exception as e:
            # grab the full stack trace
            error_traceback = traceback.format_exc()

            prompt_fix = prompt_fix_this_code(pyro_code, error_traceback)

            message_history.append({
                "role": "user",
                "content": prompt_fix
            })

            code_fix = openai_prompt(message_history, client)

            pyro_code = extract_pyro_code(code_fix)

            attempt += 1
            print(f"Attempt {attempt} failed with the following error: {e}")

    print("Max retries exceeded. The Pyro model could not be executed successfully.")
    print("Final Pyro code:")
    print(pyro_code)
    return None, max_retries

def sample_from_trace(pyro_code, n=1000):
    samples = []
    local_dict = {}
    local_dict.update({
            'pyro': pyro,
            'dist': dist,
            'torch': torch
        })
    exec(pyro_code, globals(), local_dict)
    model = local_dict['model']
    pyro.clear_param_store()
    for _ in range(n):
        # pyro.clear_param_store()
        trace = pyro.poutine.trace(model).get_trace()
        sample = {name: trace.nodes[name]['value'].item() for name in trace.nodes if trace.nodes[name]['type'] == 'sample'}
        samples.append(sample)
    return pd.DataFrame(samples)

## Plotting

# vis code for networkx dag
def plot_dag_layout(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("graph is not a DAG")

    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax)
    ax.set_title("SCM generated by LLM (topological order)")
    fig.tight_layout()
    plt.show()

