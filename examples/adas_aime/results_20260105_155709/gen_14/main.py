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
        temperature=0.5,  # Set a base temperature for moderate creativity
    ):
        self.output_format_instructions = "On the final line, output only the digits of the answer (0‑999). Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        self.query_llm = query_llm
        self.temperature = temperature
        self.max_attempts = 10  # Adjust max attempts as necessary

    def forward(self, problem: str) -> tuple[str, float]:
        """Queries the LLM with a math problem using an adaptive reflection mechanism."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)
        response, cost = None, None

        for attempt in range(self.max_attempts):
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.adjust_temperature(attempt),
            )
            if self.is_valid_response(response):
                break  # Accept the first valid response

            # Create new prompt for reflection
            task_prompt = self.refine_prompt(response)

        return response.strip(), cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format_instructions}:\n\nPlease solve the following problem step-by-step, verifying your logic after each step:\n\n{problem}\n\n"
        return system_prompt, task_prompt

    def adjust_temperature(self, attempt: int) -> float:
        """Adjust temperature based on the number of attempts made."""
        return max(0.0, 0.5 - (0.05 * attempt))

    def is_valid_response(self, response: str) -> bool:
        """Check if the response meets the formatting or correctness criteria."""
        match = re.match(r'^\d{1,3}$', response)
        return bool(match)

    def refine_prompt(self, last_response: str) -> str:
        """Create a refined prompt based on the last response."""
        return f"In the previous attempt, you provided the answer '{last_response}'. Please re-evaluate your reasoning and state if you stand by that answer, or provide a new one with clearer justification."
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
