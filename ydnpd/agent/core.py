import re
import sys
import traceback
from datetime import datetime
from collections import Counter
from pprint import pprint
from enum import Enum

from statemachine import StateMachine
from statemachine.states import States
from openai import OpenAI
import weave

from ydnpd.agent.utils import metadata_to_pandera_schema


OPENAI_KEY = 'sk-proj-qa3W3yKyqgIqIr8YXHZOT3BlbkFJNLB17J7qTKF4rrdVfLDt'
CLIENT = OpenAI(api_key=OPENAI_KEY)

MAX_ATTEMPTS = 8


class LLMSession:
    def __init__(self, specification, metadata,
                 llm_name="gpt-4o-mini",
                 llm_temperature=0.7,
                 llm_max_tokens=10000,
                 llm_top_p=1,
                 llm_frequency_penalty=0,
                 llm_presence_penalty=0,
                 verbose=False):

        self.specification = specification

        self.context = {
            "metadata": metadata,
            "last_check_info": None,
            "pandera_schema": metadata_to_pandera_schema(metadata["schema"]),
        }

        self.llm_params = {
            "model": llm_name,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "top_p": llm_top_p,
            "frequency_penalty": llm_frequency_penalty,
            "presence_penalty": llm_presence_penalty,
        }

        self.message_history = [
            {
                "role": "system",
                "content": f"You are an expert on {metadata['domain']}."
            }
        ]

        self.attempts = Counter()

        self.verbose = verbose
        self.last_transition_time = datetime.now()

    def chat_complete(self, user_message, with_answer=True):

        self.message_history.append({
            "role": "user",
            "content": user_message
            })

        if self.verbose:
            pprint(f"USER: {user_message}")

        # try:
        response = CLIENT.chat.completions.create(
            messages=self.message_history,
            **self.llm_params
        )

        assistant_message = response.choices[0].message.content

        self.message_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        if with_answer and (match := re.search(r"<Answer>(.*?)</Answer>",
                                               assistant_message,
                                               re.DOTALL)):

            answer = match.group(1)
            # answer = [item.strip() for item in match.group(1).split(",")]

            # answer_message = f"\n Extracted answer from response: {answer}\n\n"

            # self.message_history.append({
            #     "role": "assistant",
            #     "content": answer_message
            # })

            # TODO: DOES IT MESS THINGS UP?
            if self.verbose:    
                pprint(f"ASSITANT: {assistant_message}")

        else:
            answer = None

        # except Exception as e:
        #     print(f"An error occurred: {e}")
        #     return None

        return assistant_message, answer

    def execute_step(self, instruction_fn, process_fn, check_fn):

        prompt = instruction_fn(self.context)

        if prompt:
            _, answer = self.chat_complete(prompt)
        else:
            answer = None

        try:
            additional_context = process_fn(answer, self.context)

            if self.verbose:
                pprint(f"{additional_context=}")

        except ValueError:
            check_result = False
            check_info = traceback.format_exc(*sys.exc_info(), limit=1)

        else:
            (check_result,
             check_info) = check_fn(answer,
                                    additional_context,
                                    self.context)

            if check_result:
                self.context |= additional_context

        finally:
            self.context["last_check_info"] = check_info

        return check_result


class StepMixIn:

    @weave.op
    def on_enter_state(self, event, state):

        if self.model.attempts[state.id] > MAX_ATTEMPTS:
            raise RuntimeError(f"Reached max attempts on state {state.id} ({self.model.attempts[state.id]} > {MAX_ATTEMPTS})")

        self.model.attempts[state.id] += 1

        check_result = self.model.execute_step(self.model.specification[event]["instruction_fn"],
                                               self.model.specification[state.id]["processing_fn"],
                                               self.model.specification[state.id]["check_fn"])

        if state.final:
            weave.publish(self.model.context, "context")
            return

        follow_event_name = f"{state.id}_{'success' if check_result else 'failed'}"
        follow_event = getattr(self, follow_event_name)
        follow_event()


class CasualModelingAgentStage(Enum):
    SCHEMA = 1
    ELICIT_CONSTRAINTS = 2
    ROOT_NODES = 3
    ROOT_TO_NON_EDGES = 4
    NON_TO_NON_EDGES = 5
    DAG = 6
    STRUCTURAL_EQUATIONS = 7
    PARAMETERS = 8
    PYRO_CODE = 9
    ENFORCE_RANGE = 10
    ENFORCE_CONTRAINTS = 11
    FINITO = 12


class CasualModelingAgentMachine(StateMachine, StepMixIn):

    states = States.from_enum(
        CasualModelingAgentStage,
        initial=CasualModelingAgentStage.SCHEMA,
        final=CasualModelingAgentStage.FINITO,
        use_enum_instance=True,
    )

    # TODO: Refactor with dynamic generation
    SCHEMA_failed = states.SCHEMA.to.itself()
    SCHEMA_success = states.SCHEMA.to(states.ELICIT_CONSTRAINTS)

    ELICIT_CONSTRAINTS_failed = states.ELICIT_CONSTRAINTS.to.itself()
    ELICIT_CONSTRAINTS_success = states.ELICIT_CONSTRAINTS.to(states.ROOT_NODES)

    ROOT_NODES_failed = states.ROOT_NODES.to.itself()
    ROOT_NODES_success = states.ROOT_NODES.to(states.ROOT_TO_NON_EDGES)

    ROOT_TO_NON_EDGES_failed = states.ROOT_TO_NON_EDGES.to.itself()
    ROOT_TO_NON_EDGES_success = states.ROOT_TO_NON_EDGES.to(states.NON_TO_NON_EDGES)

    NON_TO_NON_EDGES_failed = states.NON_TO_NON_EDGES.to.itself()
    NON_TO_NON_EDGES_success = states.NON_TO_NON_EDGES.to(states.DAG)

    DAG_failed = states.DAG.to.itself()
    DAG_success = states.DAG.to(states.STRUCTURAL_EQUATIONS)

    STRUCTURAL_EQUATIONS_failed = states.STRUCTURAL_EQUATIONS.to.itself()
    STRUCTURAL_EQUATIONS_success = states.STRUCTURAL_EQUATIONS.to(states.PARAMETERS)

    PARAMETERS_failed = states.PARAMETERS.to.itself()
    PARAMETERS_success = states.PARAMETERS.to(states.PYRO_CODE)

    PYRO_CODE_failed = states.PYRO_CODE.to.itself()
    PYRO_CODE_success = states.PYRO_CODE.to(states.ENFORCE_RANGE)

    ENFORCE_RANGE_failed = states.ENFORCE_RANGE.to.itself()
    ENFORCE_RANGE_success = states.ENFORCE_RANGE.to(states.ENFORCE_CONTRAINTS)

    ENFORCE_CONTRAINTS_failed = states.ENFORCE_CONTRAINTS.to.itself()
    ENFORCE_CONTRAINTS_success = states.ENFORCE_CONTRAINTS.to(states.FINITO)
