import re
import traceback
from collections import Counter, defaultdict
from pprint import pprint

from openai import OpenAI

import wandb
import time
from datetime import datetime


OPENAI_KEY = 'sk-proj-qa3W3yKyqgIqIr8YXHZOT3BlbkFJNLB17J7qTKF4rrdVfLDt'
CLIENT = OpenAI(api_key=OPENAI_KEY)


class LLMSession:
    def __init__(self, specification, domain, schema,
                 llm_name="gpt-4o-mini",
                 llm_temperature=0.7,
                 llm_max_tokens=4095,
                 llm_top_p=1,
                 llm_frequency_penalty=0,
                 llm_presence_penalty=0,
                 verbose=False):

        self.specification = specification

        self.context = {
            "domain": domain,
            "schema": schema,
            "last_check_info": None
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
                "content": "You are an expert on demographic and census data."
            }
        ]

        self.attempts = Counter()

        self.verbose = verbose
        self.last_transition_time = datetime.now()

    def log_transition(self, state, result):
        current_time = datetime.now()
        duration = (current_time - self.last_transition_time).total_seconds()
        self.last_transition_time = current_time
        wandb.log({"state": state, "result": result, "duration_since_last_transition": duration})

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

    def call(self, instruction_fn, process_fn, check_fn):

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
            check_info = traceback.format_exc()
            wandb.log({"warning": f"Check failed for state with info: {check_info}"})
        else:
            (check_result,
             check_info) = check_fn(answer,
                                    additional_context,
                                    self.context)

            if check_result:
                self.context |= additional_context

        finally:
            self.context["last_check_info"] = check_info
        
        self.log_transition(state=self.context.get('current_state', 'unknown'), result=check_result)
        return check_result
