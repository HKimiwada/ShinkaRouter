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
        # Determine complexity of the problem based on keywords
        complexity_keywords = ["complex", "difficult", "challenging"]
        self.temperature = 0.7 if any(keyword in problem.lower() for keyword in complexity_keywords) else 0.0

        system_prompt, task_prompt = self.get_prompt_for_task(problem)
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        
        # Multi-step reflection for verification
        for _ in range(2):  # Allow two iterations of reflection
            reflection_prompt = (
                f"Please review your calculations and reasoning for the problem:\n\n"
                f"{problem}\n\n"
                f"Your answer is: {response}\n\n"
                f"Are there any mistakes or areas for improvement? Please explain."
            )
            reflection_response, _ = self.query_llm(
                prompt=reflection_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            if "mistake" in reflection_response.lower():
                response = "Reflection indicates an error in reasoning."
                break
        return response, cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = (
            f"{self.output_format_instructions}:\n\n"
            f"Please solve the following problem step-by-step, showing all your reasoning. "
            f"After each step, summarize your findings and check if they align with the problem requirements:\n\n"
            f"{problem}\n\n"
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