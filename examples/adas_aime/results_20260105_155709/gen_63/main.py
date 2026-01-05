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
            "On the final line output only the digits of the answer (0-999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command. "
            "Additionally, summarize the steps taken to reach this conclusion."
        )
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> tuple[str, float]:
        """Queries the LLM with a math problem."""
        system_prompt, task_prompt = self.get_prompt_for_task(problem)
        response, cost = None, None
        complexity_keywords = ["complex", "difficult", "challenging"]
        complexity_score = sum(keyword in problem.lower() for keyword in complexity_keywords)
        self.temperature = 0.0  # Default temperature
        if complexity_score >= 3:  # High complexity
            self.temperature = 1.0
        elif complexity_score == 2:  # Medium complexity
            self.temperature = 0.7
        elif complexity_score == 1:  # Low complexity
            self.temperature = 0.3
        for attempt in range(2):  # Limit to 2 attempts for generation
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            responses.append(response)
            costs.append(cost)

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
            task_prompt = f"{self.output_format_instructions}:\n\n{problem}\n\nReflection: {reflection_response}\n"
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )
            responses.append(response)
            costs.append(cost)
            for temp in [0.0, 0.5, 1.0]:
                responses = []
        costs = []
                    for temp in [0.0, 0.5, 1.0]:  # Generate responses with different temperatures
                        self.temperature = temp
            response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
            responses.append(response)
            costs.append(cost)

        # Select the best response based on a simple scoring mechanism
        final_answer = max(set(responses), key=responses.count)
        total_cost = sum(costs)
        return final_answer, total_cost

    def get_prompt_for_task(self, problem: str) -> tuple[str, str]:
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format_instructions}:\n\nPlease solve the following problem step-by-step, showing all your reasoning. After each step, summarize your findings and check if they align with the problem requirements:\n\n{problem}\n\n"
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