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
            temperature=self.temperature,
        )
            responses.append(response)
            costs.append(cost)

        verification_prompt = (
            "Please review your answer below and assess whether it's logical and correctly formatted:\n"
            f"{combined_response}\n"
            "Are there any errors in your reasoning or format? If so, provide the corrected version."
        )
        verification_response, _ = self.query_llm(
            prompt=verification_prompt,
            system=system_prompt,
            temperature=0.0
        )
        return verification_response, total_cost
        total_cost = sum(costs)
        return combined_response, total_cost

    def ensemble_responses(self, responses: List[str]) -> str:
        # Simple approach for ensembling: majority vote or choose the most complete one
        response_counter = Counter(responses)
        return response_counter.most_common(1)[0][0]

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = (
            f"{self.output_format_instructions}:\n\n"
            f"To solve the following problem, please reason through each step:\n"
            f"{problem}\n\n"
            "Explain your thought process step-by-step:\n\n"
        )
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