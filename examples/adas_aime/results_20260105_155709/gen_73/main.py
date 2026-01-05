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
        """Queries the LLM with a math problem and applies multi-step reflection."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)

        # Step 1: Initial response
        # Step 1: Initial response
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # Step 2: Perform reflection on the initial response
        reflection_prompt = (
            f"Please review your answer to the following problem and improve it if needed:\n"
            f"{problem}\nYour answer was: {response}\n"
        )
        improved_response, _ = self.query_llm(
            prompt=reflection_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # Step 3: Validate and potentially refine further based on verification
        verify_prompt = (
            "Assess the accuracy of the following solution and suggest corrections if necessary:\n"
            f"{improved_response}\n"
        )
        verification_response, _ = self.query_llm(
            prompt=verify_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )

        # Check for correctness and adjust if needed
        if "incorrect" in verification_response.lower():
            refinement_prompt = (
                "Based on your review, revise your answer with detailed reasoning:\n"
                f"Your revised answer is: {improved_response}\n"
            )
            refined_response, _ = self.query_llm(
                prompt=refinement_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            return refined_response, cost

        return improved_response, cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format_instructions}:\n\n{problem}\n\n"
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