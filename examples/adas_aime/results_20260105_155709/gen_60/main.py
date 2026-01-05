"""Agent design evaluation on math tasks."""

import re
from typing import Callable, List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from math_eval import agent_evaluation


# EVOLVE-BLOCK-START
class Agent:
    def __init__(self, query_llm: Callable, temperature=0.0):
        self.output_format_instructions = (
            "On the final line output only the digits of the answer (0-999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> Tuple[str, float]:
        """Queries the LLM with a math problem and performs enhanced multi-step reasoning."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)

        # Step 1: Get improved response using several iterations
        responses = []
        cost = 0

        for _ in range(2):  # Multiple iterations for refining the response
            response, curr_cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            responses.append(response)
            cost += curr_cost
            
            # Prepare the reflection prompt for improved reasoning
            reflection_prompt = (
                f"Review the following answer:\n{response}\n"
                f"Are there any mistakes? If so, correct the mistakes."
            )
            reflection_response, _ = self.query_llm(
                prompt=reflection_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            # Update the task prompt for the next iteration
            task_prompt = f"{self.output_format_instructions}:\n\n{problem}\n\nReflection: {reflection_response}\n"

        # Use the most common response as the final answer
        final_answer = max(set(responses), key=responses.count)
        return final_answer, cost

    def get_prompt_for_task(self, problem: str) -> Tuple[str, str]:
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
