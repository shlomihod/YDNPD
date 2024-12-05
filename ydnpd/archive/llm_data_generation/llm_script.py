import uuid
import time

from llm_utils import *
from llm_schema_config import *
from llm_prompts import *

from tqdm import tqdm

from openai import OpenAI
key = 'sk-proj-qa3W3yKyqgIqIr8YXHZOT3BlbkFJNLB17J7qTKF4rrdVfLDt'

client = OpenAI(
    api_key=key
)

results_list = []
reps = 50
# tqdm for this loop
for i in tqdm(range(reps)):
    # start time
    start_time = time.time()

    def track_run_statistics(df_id,runtime, attempts, edges, relationships, missing_variables, final_pyro_code, success, error_message=None):
        result = {
            'rep': i,
            'run_id': df_id,
            'runtime': runtime,
            'attempts': attempts,
            'edges': edges,
            'relationships': relationships,
            'missing_variables': missing_variables,
            'pyro_code': final_pyro_code,
            'success': success,
            'error_message': error_message
        }
        return result
    
    try:
        message_history = CENSUS_PROMPT.copy()
        message_history, _ = general_prompt(prompt_step_1, message_history=message_history, client=client, args={'schema': SCHEMA})
        message_history, root_rels = general_prompt(prompt_step_2, message_history=message_history, client=client, args={})
        message_history, non_root_rels = general_prompt(prompt_step_3, message_history=message_history, client=client, args={})
        relationships = root_rels + non_root_rels
        if contains_cycles(relationships):
            message_history, relationships = general_prompt(prompt_remove_cycles, message_history=message_history, client=client, args={'relationships': relationships})

        G = nx.DiGraph()
        for relationship in relationships:
            source, target = map(str.strip, relationship.split("->"))
            G.add_edge(source, target)

        assert nx.is_directed_acyclic_graph(G), "the graph is not a DAG - contains cycles"

        print("DAG relationships:")
        for edge in G.edges:
            print(f"{edge[0]} -> {edge[1]}")

        plot_dag_layout(G)
        edges = list(G.edges)
        schema_variables = set(SCHEMA.keys())
        dag_variables = set(G.nodes)
        missing_variables = schema_variables - dag_variables

        assert len(missing_variables) == 0, f"The following variables are missing from the DAG: {missing_variables}"

        message_history, pseudocode = general_prompt(prompt_step_4, message_history=message_history, client=client, args={})
        message_history, pyro_code = general_prompt(prompt_generate_pyro_code, message_history=message_history, client=client, args={'pseudocode':pseudocode}, extract_answer=extract_pyro_code)
        final_pyro_code, attempts = robust_retrieve_pyro_model(pyro_code, prompt_fix_this_code=prompt_fix_code, message_history=message_history, client=client)

        try:
            df = sample_from_trace(final_pyro_code)
            success = True
            error_message = None
        except Exception as e:
            print("Error sampling from the Pyro model:")
            print(e)
            print("Final Pyro code:")
            print(final_pyro_code)
            print("Fix Attempts:")
            print(attempts)
            success = False
            error_message = str(e)

        # store df as a csv file with an id
        df_id = str(uuid.uuid4()).split("-")[0]
        if success:
            df.to_csv(f"generated_data/{df_id}.csv", index=False)

        # end time
        end_time = time.time()
        runtime = end_time - start_time

        run_statistics = track_run_statistics(df_id, runtime, attempts, edges, relationships, list(missing_variables), final_pyro_code, success, error_message)
        results_list.append(run_statistics)
    except:
        print("Error in run")
        results_df = pd.DataFrame(results_list)
        results_df.to_pickle('run_statistics.pkl')

results_df = pd.DataFrame(results_list)
results_df.to_pickle('run_statistics.pkl')





