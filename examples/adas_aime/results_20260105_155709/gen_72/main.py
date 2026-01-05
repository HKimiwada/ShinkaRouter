"""Agent design evaluation on math tasks."""

import re
from typing import Callable, List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from math_eval import agent_evaluation


# EVOLVE-BLOCK-START
class Agent:
    def __init__(self, query_llm: Callable):
        self.output_format_instructions = (
            "On the final line output only the digits of the answer (0‑999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )
        self.query_llm = query_llm

    def forward(self, problem: str) -> Tuple[str, float]:
        """Queries the LLM with a math problem and performs multi-step reasoning."""
        # Determine complexity and set temperature
        complexity_keywords = ["complex", "difficult", "challenging", "geometry", "trigonometry"]
        temperature = 0.7 if any(keyword in problem.lower() for keyword in complexity_keywords) else 0.0

        # Break down the problem into manageable steps
        steps = self.break_down_problem(problem)
        total_cost = 0
        responses = []

        for step in steps:
            response, cost = self.query_llm(
                prompt=self.get_prompt_for_task(step),
                system="You are a skilled mathematician.",
                temperature=temperature,
            )
            responses.append(response)
            total_cost += cost

        final_answer = self.refine_answer(responses)
        return final_answer, total_cost

    def break_down_problem(self, problem: str) -> List[str]:
        """Break down the problem into manageable steps for detailed reasoning."""
        return [problem]  # Future versions can enhance this for complex problems

    def get_prompt_for_task(self, problem: str) -> str:
        return f"{self.output_format_instructions}:\n\n{problem}\n\n"

    def refine_answer(self, responses: List[str]) -> str:
        """Refine the final answer based on multiple responses."""
        unique_responses = list(set(responses))
        # Use the most common response as the starting point
        current_answer = max(set(responses), key=responses.count)

        for _ in range(2):  # Allow two iterations of refinement
            verification_prompt = (
                f"Please verify your answer to the following problem:\n\n"
                f"{current_answer}\n\n"
                f"Is this correct? If not, please provide corrections."
            )
            verification_response, _ = self.query_llm(
                prompt=verification_prompt,
                system="You are a skilled mathematician.",
                temperature=0.0,
            )
            if "incorrect" in verification_response.lower():
                # Update current_answer based on feedback
                current_answer = verification_response.strip()
        
        return current_answer

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
