"""Configuration for ShinkaRouter evolution experiment."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RouterConfig:
    """Configuration for the routing agent and evolution."""
    
    # Efficiency-Accuracy Tradeoff
    lambda_efficiency: float = 0.05  # Penalty per LLM call
    min_accuracy_threshold: float = 0.30  # Minimum acceptable accuracy
    
    # Primitive Parameters
    quick_temp: float = 0.7  # Temperature for quick_solve
    deep_temp: float = 0.0   # Temperature for deep_think
    verify_temp: float = 0.0  # Temperature for verify
    ensemble_size: int = 3    # Number of samples for ensemble
    
    # Evolution Parameters
    num_generations: int = 75
    num_islands: int = 8
    archive_size: int = 100
    max_parallel_jobs: int = 4
    
    # Evaluation Parameters
    num_experiment_runs: int = 3  # Runs per generation
    max_calls_per_problem: int = 10
    test_year: int = 2024  # Year for evolution
    holdout_year: int = 2025  # Year for final evaluation
    
    # Model Selection
    llm_models: list = None
    meta_llm_models: list = None
    
    def __post_init__(self):
        if self.llm_models is None:
            self.llm_models = ["gpt-4o-mini"]
        if self.meta_llm_models is None:
            self.meta_llm_models = ["gpt-4o-mini"]


# Global config instance
ROUTER_CONFIG = RouterConfig()


def get_lambda_schedule(generation: int) -> float:
    """
    Adaptive lambda schedule - gradually increase efficiency pressure.
    
    Args:
        generation: Current generation number
        
    Returns:
        Lambda value for this generation
    """
    schedule = {
        0: 0.0,      # First 10 gens: pure accuracy
        10: 0.01,    # Start adding efficiency pressure
        20: 0.02,
        30: 0.03,
        40: 0.05,
        50: 0.07,
        60: 0.10,
    }
    
    # Find the largest key <= generation
    lambda_val = 0.0
    for gen_threshold, val in sorted(schedule.items()):
        if generation >= gen_threshold:
            lambda_val = val
        else:
            break
    
    return lambda_val