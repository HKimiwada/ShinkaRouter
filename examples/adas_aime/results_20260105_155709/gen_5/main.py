"""Agent design evaluation on math tasks."""

import re
from typing import Callable, List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from math_eval import agent_evaluation


# EVOLVE-BLOCK-START
class Agent:
    def __init__(
        self,
        query_llm: Callable,
        temperature=0.0,
    ):
        self.output_format_instructions = "On the final line output only the digits of the answer (0‑999). Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> tuple[str, float]:
        """Queries the LLM with a math problem."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)
        responses = []
        costs = []

        for temp in [0.0, 0.5, 1.0]:
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=temp,
            )
            responses.append(response)
            costs.append(cost)

        # Combine responses (e.g., averaging values if numerical or choosing the most common if categorical)
        final_response = max(set(responses), key=responses.count)
        total_cost = sum(costs)

        return final_response, total_cost
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        verification_prompt = f"Verify the solution to the following problem: {problem}\n\nProposed solution: {response}\n\nIs it correct? If not, provide a corrected solution."
        verification_response, verification_cost = self.query_llm(
            prompt=verification_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        if "not correct" in verification_response:
            return verification_response, cost + verification_cost

        return response, cost + verification_cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"You are required to provide a detailed solution by reasoning through the problem step-by-step. {self.output_format_instructions}:\n\n{problem}\n\n"
        return system_prompt, task_prompt


# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    from utils import query_llm, create_call_limited_query_llm
    from functools import partial

    # Create base query_llm function
    base_query_llm = partial(query_llm, model_name=kwargs["model_name"])

    # Wrap it with call limiting (max 10 calls per forward pass)
    limited_query_llm = create_call_limited_query_llm(
        base_query_llm,
        max_calls=kwargs["max_calls"],
    )

    accuracy, cost_total, processed, num_llm_calls, df = agent_evaluation(
        Agent, limited_query_llm, year=kwargs["year"]
    )
    return accuracy, cost_total, processed, num_llm_calls, df