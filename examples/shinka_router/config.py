"""Configuration for ShinkaRouter evolution experiment.

SCORE SCALE:
The combined_score can be computed in two scales:
- "baseline": Matches adas_aime (score ≈ 0-100 range, accuracy - penalty)
- "normalized": Normalized scale (score ≈ 0-1 range, accuracy/100 - penalty)

Set via environment variable SHINKA_SCORE_SCALE or in RouterConfig.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RouterConfig:
    """Configuration for the routing agent and evolution."""
    
    # ========================================================================
    # Score Configuration
    # ========================================================================
    
    # Score scale: "baseline" (0-100) or "normalized" (0-1)
    # "baseline" recommended for compatibility with ShinkaEvolve defaults
    score_scale: str = "baseline"
    
    # Efficiency-Accuracy Tradeoff
    # Note: lambda values are scaled appropriately for each score_scale
    lambda_efficiency: float = 0.05  # Base penalty per LLM call
    min_accuracy_threshold: float = 0.30  # Minimum acceptable accuracy (30%)
    
    # ========================================================================
    # Primitive Parameters
    # ========================================================================
    
    quick_temp: float = 0.7   # Temperature for quick_solve (exploration)
    deep_temp: float = 0.0    # Temperature for deep_think (deterministic)
    verify_temp: float = 0.0  # Temperature for verify (deterministic)
    ensemble_size: int = 3    # Default samples for ensemble voting
    
    # ========================================================================
    # Evolution Parameters
    # ========================================================================
    
    num_generations: int = 75
    num_islands: int = 8
    archive_size: int = 100
    max_parallel_jobs: int = 4
    
    # ========================================================================
    # Evaluation Parameters
    # ========================================================================
    
    num_experiment_runs: int = 3   # Runs per candidate evaluation
    max_calls_per_problem: int = 10  # Hard limit on LLM calls
    test_year: int = 2024          # Year for evolution (training)
    holdout_year: int = 2025       # Year for final evaluation (test)
    
    # ========================================================================
    # Model Selection
    # ========================================================================
    
    llm_models: Optional[List[str]] = None
    meta_llm_models: Optional[List[str]] = None
    eval_model: str = "gpt-4o-mini"  # Model used for agent evaluation
    
    def __post_init__(self):
        if self.llm_models is None:
            self.llm_models = ["gpt-4o-mini"]
        if self.meta_llm_models is None:
            self.meta_llm_models = ["gpt-4o-mini"]
        
        # Apply environment variable override for score scale
        env_scale = os.environ.get("SHINKA_SCORE_SCALE")
        if env_scale:
            self.score_scale = env_scale


# Global config instance
ROUTER_CONFIG = RouterConfig()


def get_lambda_schedule(generation: int) -> float:
    """
    Adaptive lambda schedule - gradually increase efficiency pressure.
    
    The curriculum approach:
    1. Early generations (0-9): Focus on accuracy only (λ=0)
    2. Mid generations (10-39): Gradually add efficiency pressure
    3. Late generations (40+): Strong efficiency optimization
    
    Args:
        generation: Current generation number
        
    Returns:
        Lambda value for this generation
    
    Note:
        Lambda values are designed for the "baseline" score scale.
        When using "normalized" scale, the effective penalty is the same
        because it's applied differently in compute_efficiency_score().
    """
    schedule = {
        0: 0.0,      # Gen 0-9: Pure accuracy (no efficiency penalty)
        10: 0.01,    # Gen 10-19: Slight efficiency pressure
        20: 0.02,    # Gen 20-29: Light pressure
        30: 0.03,    # Gen 30-39: Moderate pressure
        40: 0.05,    # Gen 40-49: Significant pressure
        50: 0.07,    # Gen 50-59: Strong pressure
        60: 0.10,    # Gen 60+: Maximum pressure
    }
    
    # Find the largest key <= generation
    lambda_val = 0.0
    for gen_threshold, val in sorted(schedule.items()):
        if generation >= gen_threshold:
            lambda_val = val
        else:
            break
    
    return lambda_val


def get_lambda_for_scale(generation: int, score_scale: str = "baseline") -> float:
    """
    Get lambda adjusted for score scale.
    
    Args:
        generation: Current generation
        score_scale: "baseline" or "normalized"
        
    Returns:
        Lambda value (same for both scales, applied differently in scoring)
    """
    return get_lambda_schedule(generation)


# ============================================================================
# Primitive Documentation
# ============================================================================

PRIMITIVE_DOCS = """
Available Primitives for Routing:

1. quick_solve(problem) -> (response, cost)
   - Fast solving with temperature=0.7
   - Good for: Easy problems, first-pass attempts
   - Typical calls: 1
   
2. deep_think(problem) -> (response, cost)
   - Chain-of-thought reasoning with temperature=0.0
   - Good for: Complex multi-step problems
   - Typical calls: 1

3. verify(problem, candidate_answer) -> (response, cost)
   - Skeptical verification of a proposed answer
   - Good for: Validating uncertain solutions
   - Typical calls: 1

4. python_calc(problem) -> (response, cost)
   - Systematic step-by-step calculation
   - Good for: Computation-heavy problems
   - Typical calls: 1

5. ensemble_vote(problem, n=3) -> (response, cost)
   - Generate n solutions, majority vote on answer
   - Good for: Medium difficulty, diverse solutions
   - Typical calls: n (default 3)

6. self_critique(problem, draft_response) -> (response, cost)
   - Review and improve a draft solution
   - Good for: Iterative refinement
   - Typical calls: 1

7. estimate_difficulty(problem) -> (difficulty, cost)
   - Returns 'easy', 'medium', or 'hard'
   - Good for: Adaptive routing decisions
   - Typical calls: 1

8. classify_problem_type(problem) -> (type, cost)
   - Returns problem category (algebra, geometry, etc.)
   - Good for: Type-specialized routing
   - Typical calls: 1

Routing Strategies:
- Difficulty-based: estimate_difficulty -> route to appropriate primitive
- Sequential: quick_solve -> verify if uncertain
- Ensemble: ensemble_vote for medium-difficulty
- Type-based: classify_problem_type -> specialized primitive
- Multi-stage: quick_solve -> self_critique -> verify (uses 3 calls)
"""