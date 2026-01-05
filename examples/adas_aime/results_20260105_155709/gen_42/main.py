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
        """Queries the LLM with a math problem and applies multi-step reasoning."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)

        # Step 1: Initial response
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # Step 2: Reflection and refinement
        reflection_prompt = (
            f"Please review your answer to the following problem:\n\n"
            f"{problem}\n\n"
            f"Your answer is: {response}\n\n"
            f"Confirm if the answer is correct, and if not, explain why. Provide the correct answer if possible."
        )
        verification_response, _ = self.query_llm(
            prompt=reflection_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # If the verification indicates an error, revise the answer
        if "incorrect" in verification_response.lower():
            refinement_prompt = (
                "Based on the feedback, please revise and improve the previous answer with detailed reasoning."
            )
            refined_response, _ = self.query_llm(
                prompt=refinement_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            return refined_response, cost

        return response, cost

    def get_prompt_for_task(self, problem: str) -> Tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = (
            f"{self.output_format_instructions}:\n\n"
            f"To solve the math problem step-by-step, please think through the following:\n"
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
