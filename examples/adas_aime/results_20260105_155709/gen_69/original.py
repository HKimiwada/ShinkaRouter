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
        self.output_format_instructions = (
            "On the final line output only the digits of the answer (0‑999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> Tuple[str, float]:
        """Queries the LLM with a math problem and applies multi-step reflection."""
        steps = self.break_down_problem(problem)
        responses = []
        total_cost = 0

        for step in steps:
            system_prompt, task_prompt = self.get_prompt_for_task(step)
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            responses.append(response)
            total_cost += cost

        final_answer = self.refine_answer(responses)
        return final_answer, total_cost

    def break_down_problem(self, problem: str) -> List[str]:
        """Break down the problem into several steps for detailed reasoning."""
        # Here we simplify to generate steps. More advanced breakdown can be implemented.
        return [problem]  # In future, enhance with a real breakdown

    def get_prompt_for_task(self, problem: str) -> Tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format_instructions}:\n\nPlease solve the following problem step-by-step:\n\n{problem}\n\n"
        return system_prompt, task_prompt

    def refine_answer(self, responses: List[str]) -> str:
        """Refine the final answer based on multiple responses."""
        unique_responses = list(set(responses))
        # Return the most frequently occurring response (could be improved with better logic)
        return max(set(unique_responses), key=unique_responses.count)

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