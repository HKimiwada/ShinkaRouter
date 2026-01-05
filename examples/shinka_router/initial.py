"""ShinkaRouter: Agent with routing primitives for AIME problems.

This agent provides a toolbox of specialized primitives that can be
composed in different ways. The forward() method contains the routing
logic that will be evolved by ShinkaEvolve.
"""

import re
from typing import Callable, Tuple, List, Optional


class Agent:
    """Agent with multiple reasoning primitives for AIME math problems."""
    
    def __init__(
        self,
        query_llm: Callable,
        quick_temp: float = 0.7,
        deep_temp: float = 0.0,
        verify_temp: float = 0.0,
        ensemble_size: int = 3,
    ):
        """
        Initialize the routing agent.
        
        Args:
            query_llm: Function to query the LLM
            quick_temp: Temperature for quick_solve
            deep_temp: Temperature for deep_think
            verify_temp: Temperature for verify
            ensemble_size: Number of samples for ensemble voting
        """
        self.query_llm = query_llm
        self.quick_temp = quick_temp
        self.deep_temp = deep_temp
        self.verify_temp = verify_temp
        self.ensemble_size = ensemble_size
        
        # Standard output format for AIME
        self.output_format = (
            "On the final line output only the digits of the answer (0-999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )
    
    # ========================================================================
    # PRIMITIVE METHODS (Stable - Not Evolved)
    # ========================================================================
    
    def quick_solve(self, problem: str) -> Tuple[str, float]:
        """
        Fast solving with higher temperature for quick guessing.
        Good for easy problems or first-pass attempts.
        """
        system_prompt = "You are a skilled mathematician. Solve quickly and efficiently."
        task_prompt = f"{self.output_format}\n\n{problem}\n\n"
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.quick_temp,
        )
        return response, cost
    
    def deep_think(self, problem: str) -> Tuple[str, float]:
        """
        Careful reasoning with chain-of-thought and low temperature.
        Good for complex problems requiring step-by-step analysis.
        """
        system_prompt = (
            "You are an expert mathematician. Think step-by-step, "
            "showing all your reasoning before arriving at the final answer."
        )
        task_prompt = (
            f"Solve this problem carefully with detailed reasoning:\n\n"
            f"{problem}\n\n"
            f"{self.output_format}"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.deep_temp,
        )
        return response, cost
    
    def verify(self, problem: str, candidate_answer: str) -> Tuple[str, float]:
        """
        Act as a skeptical verifier checking a proposed solution.
        Returns either confirmation or a corrected answer.
        """
        system_prompt = (
            "You are a rigorous mathematics professor checking a student's answer. "
            "Verify if the answer is correct. If wrong, provide the correct answer."
        )
        task_prompt = (
            f"Problem: {problem}\n\n"
            f"Proposed Answer: {candidate_answer}\n\n"
            f"Is this answer correct? If not, what is the correct answer?\n"
            f"{self.output_format}"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.verify_temp,
        )
        return response, cost
    
    def python_calc(self, problem: str) -> Tuple[str, float]:
        """
        Extract computational aspects and solve using Python code.
        Good for problems requiring numerical calculations.
        """
        system_prompt = (
            "You are a mathematician who solves problems using Python. "
            "Write and execute Python code to solve the problem, then provide the final answer."
        )
        task_prompt = (
            f"Solve this problem using Python code:\n\n"
            f"{problem}\n\n"
            f"Show your code and calculations, then {self.output_format}"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.deep_temp,
        )
        return response, cost
    
    def ensemble_vote(self, problem: str, n: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate multiple solutions and take majority vote.
        Good for medium-difficulty problems where consensus helps.
        """
        if n is None:
            n = self.ensemble_size
        
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format}\n\n{problem}\n\n"
        
        responses = []
        total_cost = 0.0
        
        for _ in range(n):
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=0.7,  # Medium temp for diversity
            )
            responses.append(response)
            total_cost += cost
        
        # Return the first response (in practice, would vote on extracted answers)
        # The voting logic would be implemented in a real scenario
        return responses[0], total_cost
    
    def self_critique(self, problem: str, draft_answer: str) -> Tuple[str, float]:
        """
        Generate a draft solution, then critique and refine it.
        Good for complex problems benefiting from iterative refinement.
        """
        system_prompt = (
            "You are a mathematician reviewing your own work. "
            "Critique the draft solution and provide an improved answer."
        )
        task_prompt = (
            f"Problem: {problem}\n\n"
            f"Draft Solution: {draft_answer}\n\n"
            f"Review this solution carefully. If there are any errors or improvements, "
            f"provide a better solution. Otherwise, confirm the answer.\n"
            f"{self.output_format}"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.deep_temp,
        )
        return response, cost
    
    def estimate_difficulty(self, problem: str) -> Tuple[str, float]:
        """
        Estimate problem difficulty as 'easy', 'medium', or 'hard'.
        Can be used for adaptive routing decisions.
        """
        system_prompt = "You are an expert at evaluating problem difficulty."
        task_prompt = (
            f"Analyze this AIME problem and classify its difficulty as "
            f"'easy', 'medium', or 'hard':\n\n{problem}\n\n"
            f"Respond with only one word: easy, medium, or hard."
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=0.0,
        )
        
        # Extract difficulty level
        difficulty = response.strip().lower()
        if difficulty not in ['easy', 'medium', 'hard']:
            difficulty = 'medium'  # Default
        
        return difficulty, cost

    # ========================================================================
    # ROUTING LOGIC (Evolved by ShinkaEvolve)
    # ========================================================================
    
    # EVOLVE-BLOCK-START
    def forward(self, problem: str) -> Tuple[str, float]:
        """
        Main entry point - routes problem to appropriate primitives.
        This method will be evolved by ShinkaEvolve to discover
        optimal routing strategies.
        
        Initial baseline: Simple quick_solve approach.
        """
        # Baseline: Just use quick_solve for all problems
        response, cost = self.quick_solve(problem)
        return response, cost
    # EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    """
    Entry point called by evaluate.py.
    
    Args:
        model_name: Name of the LLM model to use
        year: AIME dataset year
        max_calls: Maximum LLM calls per problem
        
    Returns:
        Tuple of (accuracy, cost, processed, num_llm_calls, dataframe)
    """
    from utils import query_llm, create_call_limited_query_llm
    from functools import partial
    from config import ROUTER_CONFIG

    # Create base query_llm function
    base_query_llm = partial(query_llm, model_name=kwargs["model_name"])

    # Wrap it with call limiting
    limited_query_llm = create_call_limited_query_llm(
        base_query_llm,
        max_calls=kwargs["max_calls"],
    )

    # Import evaluation function
    from math_eval import agent_evaluation

    # Run evaluation with configured parameters
    accuracy, cost_total, processed, num_llm_calls, df = agent_evaluation(
        Agent, 
        limited_query_llm, 
        year=kwargs["year"],
        quick_temp=ROUTER_CONFIG.quick_temp,
        deep_temp=ROUTER_CONFIG.deep_temp,
        verify_temp=ROUTER_CONFIG.verify_temp,
        ensemble_size=ROUTER_CONFIG.ensemble_size,
    )
    
    return accuracy, cost_total, processed, num_llm_calls, df