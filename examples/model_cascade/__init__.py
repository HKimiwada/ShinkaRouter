"""
ShinkaRouter: Multi-model routing optimization via evolutionary algorithms.

Evolves routing logic to find optimal cost/accuracy trade-offs across
model tiers (gpt-4o-mini, gpt-4.1-nano, o4-mini).
"""

from .router import Router
from .evaluator import RouterEvaluator, Problem, RoutingMetrics
from .evolution import RouterEvolution
from .pareto import (
    plot_pareto_frontier,
    compute_pareto_frontier,
    compute_dominated_area_percentage,
)

__all__ = [
    'Router',
    'RouterEvaluator',
    'Problem',
    'RoutingMetrics',
    'RouterEvolution',
    'plot_pareto_frontier',
    'compute_pareto_frontier',
    'compute_dominated_area_percentage',
]