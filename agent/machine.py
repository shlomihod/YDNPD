import wandb
import time
from datetime import datetime

from enum import Enum

from statemachine import StateMachine
from statemachine.states import States


MAX_ATTEMPTS = 8

PROMPT_SUFFIX = "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) within the tags <Answer>...</Answer>, separated by \", \". "

PYRO_FAILED_CODE_TEMPLATE = """
            Pyro code validation failed, please review and try again.
            Here is the Traceback:
            ```
            {context['last_check_info']}
            ```
            ALWAYS return Pyro code, ready to be execute without Markdown formattig, within the tags <Answer>...</Answer>.
            """

class StepMixIn:

    def on_enter_state(self, event, state):
        start_time = time.time()
        wandb.log({"state": state.id, "event": event, "attempt": self.model.attempts[state.id] + 1})

        if self.model.attempts[state.id] > MAX_ATTEMPTS:
            wandb.log({"error": f"Reached max attempts on state {state.id}"})
            raise RuntimeError(f"Reached max attempts on state {state.id} ({self.model.attempts[state.id]} > {MAX_ATTEMPTS})")

        self.model.attempts[state.id] += 1

        check_result = self.model.call(self.model.specification[event]["instruction_fn"],
                                       self.model.specification[state.id]["processing_fn"],
                                       self.model.specification[state.id]["check_fn"])

        end_time = time.time()
        duration = end_time - start_time
        wandb.log({
            "state": state.id,
            "result": "success" if check_result else "failure",
            "duration": duration
        })

        if state.final:
            wandb.log({"final_state": state.id})
            return

        follow_event_name = f"{state.id}_{'success' if check_result else 'failed'}"
        follow_event = getattr(self, follow_event_name)
        follow_event()


class CasualModelingAgentStage(Enum):
    SCHEME = 1
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
        initial=CasualModelingAgentStage.SCHEME,
        final=CasualModelingAgentStage.FINITO,
        use_enum_instance=True,
    )

    # TODO: Refactor with dynamic generation
    SCHEME_failed = states.SCHEME.to.itself()
    SCHEME_success = states.SCHEME.to(states.ELICIT_CONSTRAINTS)

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
