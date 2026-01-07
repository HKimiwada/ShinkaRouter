"""ShinkaRouter: Agent with routing primitives for AIME problems.

This agent provides a toolbox of specialized primitives that can be
composed in different ways. The forward() method contains the routing
logic that will be evolved by ShinkaEvolve.

FIXES from original:
1. verify() now accepts full response, not just answer
2. deep_think() uses same prompt structure as baseline (format first)
3. ensemble_vote() uses lower temperature (0.5 instead of 0.7)
4. All primitives use consistent prompt structure
"""

import re
from typing import Callable, Tuple, List, Optional, Dict
from collections import Counter


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
        
        # Primitive call tracking (reset per forward() call)
        self._primitive_calls: List[str] = []
    
    def _track_call(self, primitive_name: str) -> None:
        """Track a primitive call for analysis."""
        self._primitive_calls.append(primitive_name)
    
    def get_primitive_calls(self) -> List[str]:
        """Return list of primitives called in this forward pass."""
        return self._primitive_calls.copy()
    
    def reset_tracking(self) -> None:
        """Reset primitive call tracking."""
        self._primitive_calls = []
    
    @staticmethod
    def extract_boxed_answer(response: str) -> Optional[str]:
        """Extract answer from \\boxed{} in response."""
        idx = response.rfind("\\boxed")
        if idx < 0:
            idx = response.rfind("\\fbox")
        if idx < 0:
            return None
        
        brace_idx = response.find("{", idx)
        if brace_idx < 0:
            return None
        
        level = 0
        for i in range(brace_idx, len(response)):
            if response[i] == "{":
                level += 1
            elif response[i] == "}":
                level -= 1
                if level == 0:
                    content = response[brace_idx + 1:i]
                    # Clean and normalize
                    content = content.strip().lstrip("0") or "0"
                    return content
        return None

    # ========================================================================
    # PRIMITIVE METHODS (Stable - Not Evolved)
    # ========================================================================
    
    def baseline_solve(self, problem: str) -> Tuple[str, float]:
        """
        Exact replication of adas_aime baseline behavior.
        Temperature=0.0, simple system prompt, minimal task prompt.
        Use this for fair comparison with baseline ShinkaEvolve.
        """
        self._track_call("baseline_solve")
        
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format}:\n\n{problem}\n\n"
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=0.0,
        )
        return response, cost
    
    def quick_solve(self, problem: str) -> Tuple[str, float]:
        """
        Fast solving with higher temperature for diverse attempts.
        Good for easy problems or first-pass attempts.
        Uses same prompt structure as baseline.
        """
        self._track_call("quick_solve")
        
        system_prompt = "You are a skilled mathematician."
        # Same structure as baseline - format first, then problem
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
        
        FIX: Uses same prompt structure as baseline (format first) for
        consistent answer extraction.
        """
        self._track_call("deep_think")
        
        system_prompt = (
            "You are an expert mathematician. Think step-by-step, "
            "showing all your reasoning before arriving at the final answer."
        )
        # Format instruction first (like baseline), then problem
        task_prompt = (
            f"{self.output_format}\n\n"
            f"Solve this problem carefully with detailed reasoning:\n\n"
            f"{problem}"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.deep_temp,
        )
        return response, cost
    
    def verify(self, problem: str, candidate_response: str) -> Tuple[str, float]:
        """
        Act as a skeptical verifier checking a proposed solution.
        Returns either confirmation or a corrected answer.
        
        FIX: Now accepts full response (with reasoning), not just the answer.
        This allows the verifier to actually check the work.
        
        Args:
            problem: The original problem
            candidate_response: The FULL response including reasoning (not just answer)
        """
        self._track_call("verify")
        
        system_prompt = (
            "You are a rigorous mathematics professor checking a student's work. "
            "Carefully verify the solution step by step. "
            "If you find any errors, explain them and provide the correct answer."
        )
        task_prompt = (
            f"{self.output_format}\n\n"
            f"Problem: {problem}\n\n"
            f"Student's Solution:\n{candidate_response}\n\n"
            f"Verify this solution. Check each step for errors. "
            f"Provide the correct final answer."
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.verify_temp,
        )
        return response, cost
    
    def python_calc(self, problem: str) -> Tuple[str, float]:
        """
        Prompt LLM to reason through computation step-by-step.
        Encourages systematic calculation approach.
        """
        self._track_call("python_calc")
        
        system_prompt = (
            "You are a mathematician who solves problems systematically. "
            "Break down any calculations step-by-step, showing intermediate results. "
            "Double-check arithmetic by computing in multiple ways if helpful."
        )
        # Format first for consistent extraction
        task_prompt = (
            f"{self.output_format}\n\n"
            f"Solve this problem with careful step-by-step calculations:\n\n"
            f"{problem}\n\n"
            f"Show all intermediate calculations clearly."
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.deep_temp,
        )
        return response, cost
    
    def ensemble_vote(self, problem: str, n: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate multiple solutions and take majority vote on the answer.
        Good for medium-difficulty problems where consensus helps.
        
        FIX: Uses moderate temperature (0.5) instead of 0.7 to reduce noise
        while still getting diversity.
        
        Returns the response containing the winning answer.
        """
        self._track_call("ensemble_vote")
        
        if n is None:
            n = self.ensemble_size
        
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format}\n\n{problem}\n\n"
        
        responses = []
        answers = []
        total_cost = 0.0
        
        # FIX: Use moderate temperature for better accuracy while maintaining diversity
        ensemble_temp = 0.5
        
        for _ in range(n):
            response, cost = self.query_llm(
                prompt=task_prompt,
                system=system_prompt,
                temperature=ensemble_temp,
            )
            responses.append(response)
            total_cost += cost
            
            # Extract answer from this response
            ans = self.extract_boxed_answer(response)
            if ans:
                answers.append(ans)
        
        # Majority vote
        if answers:
            vote_counts = Counter(answers)
            winner, count = vote_counts.most_common(1)[0]
            
            # Return response that contains the winning answer
            for resp in responses:
                extracted = self.extract_boxed_answer(resp)
                if extracted == winner:
                    return resp, total_cost
        
        # Fallback to first response if no valid answers found
        return responses[0], total_cost
    
    def self_critique(self, problem: str, draft_response: str) -> Tuple[str, float]:
        """
        Critique a draft solution and provide an improved answer.
        Good for complex problems benefiting from iterative refinement.
        
        Args:
            problem: The original problem
            draft_response: The full draft response (not just the answer)
        """
        self._track_call("self_critique")
        
        system_prompt = (
            "You are a mathematician reviewing your own work. "
            "Carefully check the solution for errors in logic, calculation, or reasoning. "
            "Provide an improved answer if you find any issues."
        )
        # Format first for consistent extraction
        task_prompt = (
            f"{self.output_format}\n\n"
            f"Problem: {problem}\n\n"
            f"Draft Solution:\n{draft_response}\n\n"
            f"Review this solution carefully. Check for:\n"
            f"1. Arithmetic errors\n"
            f"2. Logical mistakes\n"
            f"3. Missing cases\n"
            f"4. Incorrect assumptions\n\n"
            f"Provide your final answer (same or corrected)."
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
        
        Returns:
            Tuple of (difficulty_level, cost) where difficulty is 'easy', 'medium', or 'hard'
        """
        self._track_call("estimate_difficulty")
        
        system_prompt = "You are an expert at evaluating AIME problem difficulty."
        task_prompt = (
            f"Analyze this AIME problem and classify its difficulty.\n\n"
            f"Problem: {problem}\n\n"
            f"Consider:\n"
            f"- Number of steps required\n"
            f"- Complexity of mathematical concepts\n"
            f"- Amount of computation needed\n"
            f"- Whether it requires insight vs. standard techniques\n\n"
            f"Respond with exactly one word: easy, medium, or hard"
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=0.0,
        )
        
        # Extract difficulty level
        difficulty = response.strip().lower()
        # Handle common variations
        if "easy" in difficulty:
            difficulty = "easy"
        elif "hard" in difficulty:
            difficulty = "hard"
        else:
            difficulty = "medium"  # Default
        
        return difficulty, cost
    
    def classify_problem_type(self, problem: str) -> Tuple[str, float]:
        """
        Classify the problem type (algebra, geometry, number theory, etc.)
        Can be used for specialized routing.
        
        Returns:
            Tuple of (problem_type, cost)
        """
        self._track_call("classify_problem_type")
        
        system_prompt = "You are an expert at categorizing mathematics competition problems."
        task_prompt = (
            f"Classify this AIME problem into ONE primary category:\n\n"
            f"Problem: {problem}\n\n"
            f"Categories:\n"
            f"- algebra (equations, polynomials, sequences)\n"
            f"- geometry (shapes, angles, coordinates)\n"
            f"- number_theory (divisibility, primes, modular arithmetic)\n"
            f"- combinatorics (counting, probability, arrangements)\n"
            f"- calculus (limits, optimization - rare in AIME)\n\n"
            f"Respond with exactly one word from the categories above."
        )
        
        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=0.0,
        )
        
        # Normalize response
        ptype = response.strip().lower().replace(" ", "_")
        valid_types = ["algebra", "geometry", "number_theory", "combinatorics", "calculus"]
        
        for vt in valid_types:
            if vt in ptype:
                return vt, cost
        
        return "algebra", cost  # Default

    # ========================================================================
    # ROUTING LOGIC (Evolved by ShinkaEvolve)
    # ========================================================================
    
    # EVOLVE-BLOCK-START
"""ShinkaRouter: Adaptive difficulty-based routing agent for AIME problems.

This agent estimates problem difficulty and routes problems through a tailored sequence of primitives:
- Easy: Quick solve + verify
- Medium: Ensemble voting + verify
- Hard: Deep think + self critique + verify
This methodology minimizes LLM calls while maximizing solution correctness.

FIXES from previous:
1. Verify accepts full response; uses consistent prompt structure.
2. Deep_think adopts prompt structure akin to baseline.
3. Ensemble_vote uses lower temperature (0.5).
4. Modular primitives promote efficient, accurate reasoning.
"""

import re
from typing import Callable, Tuple, List
from collections import Counter

class Agent:
    def __init__(
        self,
        query_llm: Callable,
        quick_temp: float = 0.7,
        deep_temp: float = 0.0,
        verify_temp: float = 0.0,
        ensemble_size: int = 3,
    ):
        self.query_llm = query_llm
        self.quick_temp = quick_temp
        self.deep_temp = deep_temp
        self.verify_temp = verify_temp
        self.ensemble_size = ensemble_size
        self._primitive_calls: List[str] = []

        self.output_format = (
            "On the final line output only the digits of the answer (0-999). "
            "Provide your final answer enclosed in a LaTeX \\boxed{{...}} command."
        )

    def _track_call(self, name: str) -> None:
        self._primitive_calls.append(name)

    def get_primitive_calls(self) -> List[str]:
        return self._primitive_calls.copy()

    def reset_tracking(self) -> None:
        self._primitive_calls = []

    @staticmethod
    def extract_boxed_answer(response: str) -> Optional[str]:
        idx = response.rfind("\\boxed")
        if idx < 0:
            idx = response.rfind("\\fbox")
        if idx < 0:
            return None
        brace_idx = response.find("{", idx)
        if brace_idx < 0:
            return None
        level = 0
        for i in range(brace_idx, len(response)):
            if response[i] == "{":
                level += 1
            elif response[i] == "}":
                level -= 1
                if level == 0:
                    content = response[brace_idx + 1:i]
                    return content.strip().lstrip("0") or "0"
        return None

    # Primitive methods with standardized prompts
    def baseline_solve(self, problem: str) -> Tuple[str, float]:
        self._track_call("baseline_solve")
        prompt = f"{self.output_format}:\n\n{problem}\n\n"
        return self.query_llm(prompt=prompt, system="You are a skilled mathematician.", temperature=0.0)

    def quick_solve(self, problem: str) -> Tuple[str, float]:
        self._track_call("quick_solve")
        prompt = f"{self.output_format}\n\n{problem}\n\n"
        return self.query_llm(prompt=prompt, system="You are a skilled mathematician.", temperature=self.quick_temp)

    def deep_think(self, problem: str) -> Tuple[str, float]:
        self._track_call("deep_think")
        prompt = (
            f"{self.output_format}\n\n"
            f"Solve this problem carefully with detailed reasoning:\n\n{problem}"
        )
        return self.query_llm(prompt=prompt, system="You are an expert mathematician. Think step-by-step, showing all reasoning.", temperature=self.deep_temp)

    def verify(self, problem: str, response: str) -> Tuple[str, float]:
        self._track_call("verify")
        prompt = (
            f"{self.output_format}\n\n"
            f"Problem: {problem}\n\n"
            f"Solution:\n{response}\n\n"
            f"Verify this solution step-by-step. If errors are found, correct them."
        )
        return self.query_llm(prompt=prompt, system="You are a rigorous mathematics professor.", temperature=self.verify_temp)

    def ensemble_vote(self, problem: str, n: Optional[int] = None) -> Tuple[str, float]:
        self._track_call("ensemble_vote")
        n = n or self.ensemble_size
        prompt = f"{self.output_format}\n\n{problem}\n\n"
        responses = []
        total_cost = 0.0
        for _ in range(n):
            resp, cost = self.query_llm(prompt=prompt, system="You are a skilled mathematician.", temperature=0.5)
            responses.append((resp, cost))
            total_cost += cost
        answers = [self.extract_boxed_answer(r[0]) for r in responses if self.extract_boxed_answer(r[0]) is not None]
        if answers:
            count = Counter(answers)
            winner, _ = count.most_common(1)[0]
            for r, c in responses:
                if self.extract_boxed_answer(r) == winner:
                    return r, total_cost
        return responses[0][0], total_cost

    def self_critique(self, problem: str, draft_response: str) -> Tuple[str, float]:
        self._track_call("self_critique")
        prompt = (
            f"{self.output_format}\n\n"
            f"Problem: {problem}\n\n"
            f"Draft Solution:\n{draft_response}\n\n"
            f"Review for errors and provide an improved solution."
        )
        return self.query_llm(prompt=prompt, system="You are a mathematician reviewing your own work.", temperature=self.deep_temp)

    def estimate_difficulty(self, problem: str) -> Tuple[str, float]:
        self._track_call("estimate_difficulty")
        prompt = (
            f"Analyze this problem and classify as easy, medium, or hard:\n\n{problem}"
        )
        response, cost = self.query_llm(prompt=prompt, system="You are an expert at difficulty assessment.", temperature=0.0)
        diff = response.strip().lower()
        if "easy" in diff:
            return "easy", cost
        elif "hard" in diff:
            return "hard", cost
        else:
            return "medium", cost

    def classify_problem_type(self, problem: str) -> Tuple[str, float]:
        self._track_call("classify_problem_type")
        prompt = (
            f"Classify this problem into one category: algebra, geometry, number_theory, combinatorics, calculus."
        )
        response, cost = self.query_llm(prompt=prompt, system="You are an expert problem classifier.", temperature=0.0)
        ptype = response.strip().lower().replace(" ", "_")
        for category in ["algebra", "geometry", "number_theory", "combinatorics", "calculus"]:
            if category in ptype:
                return category, cost
        return "algebra", cost

    # Main routing logic with hierarchical, difficulty-based approach
    def forward(self, problem: str) -> Tuple[str, float]:
        """
        Routes problems based on estimated difficulty:
        - Easy: quick_solve + verify
        - Medium: ensemble_vote + verify
        - Hard: deep_think + self_critique + verify
        """
        self.reset_tracking()
        total_cost = 0.0

        # Step 1: Estimate difficulty
        difficulty, c_diff = self.estimate_difficulty(problem)
        total_cost += c_diff

        # Step 2: Condition routing based on difficulty
        if difficulty == "easy":
            resp, c_resp = self.quick_solve(problem)
            total_cost += c_resp
            resp, c_ver = self.verify(problem, resp)
            total_cost += c_ver
            return resp, total_cost

        elif difficulty == "medium":
            resp, c_resp = self.ensemble_vote(problem)
            total_cost += c_resp
            resp, c_ver = self.verify(problem, resp)
            total_cost += c_ver
            return resp, total_cost

        else:  # hard
            resp, c_resp = self.deep_think(problem)
            total_cost += c_resp
            # Self critique for refinement
            crit_resp, c_crit = self.self_critique(problem, resp)
            total_cost += c_crit
            # Final verification
            final_resp, c_ver2 = self.verify(problem, crit_resp)
            total_cost += c_ver2
            return final_resp, total_cost
# END OF CROSSOVER
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