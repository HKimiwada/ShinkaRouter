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
        temperature=0.3,  # A slightly broader temperature for variance
    ):
        self.output_format_instructions = (
            "On the final line output only the digits of the answer (0‑999). Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> Tuple[str, float]:
        """Queries the LLM with a math problem."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)
        
        # Initial response generation
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        
        # Reflection step: Ask the model to verify its answer
        verification_prompt = f"Please verify your answer: {response}. If incorrect, state the correct answer."
        verification_response, verification_cost = self.query_llm(
            prompt=verification_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        
        # Return the verified response without LaTeX formatting (to fetch only the answer)
        return self.extract_final_answer(verification_response), cost + verification_cost

    def get_prompt_for_task(self, problem: str) -> Tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = (
            f"{self.output_format_instructions}\n\n"
            "To solve the math problem step-by-step, please follow these tasks:\n"
            "1. Identify the key elements of the problem.\n"
            "2. State what needs to be calculated clearly.\n"
            "3. Show each step of the calculation clearly.\n"
            f"{problem}\n\n"
        )
        return system_prompt, task_prompt

    def extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the LLM response."""
        match = re.search(r'\\boxed{(\d{1,3})}', response)
        return match.group(1) if match else "0"  # Return 0 if no match

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