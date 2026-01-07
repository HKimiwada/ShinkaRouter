"""
Evaluation framework for routing policies with cost/accuracy tracking.
"""
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Represents a math problem to solve."""
    question: str
    answer: str
    difficulty: str = "unknown"
    

@dataclass
class RoutingMetrics:
    """Metrics for a router's performance."""
    accuracy: float
    total_cost: float
    avg_cost_per_problem: float
    efficiency: float  # accuracy per dollar
    model_usage: Dict[str, int]  # counts per model
    num_problems: int
    correct_count: int
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'total_cost': self.total_cost,
            'avg_cost': self.avg_cost_per_problem,
            'efficiency': self.efficiency,
            'model_usage': self.model_usage,
            'num_problems': self.num_problems,
            'correct': self.correct_count,
        }


class RouterEvaluator:
    """Evaluates routers on problem datasets."""
    
    # Cost per 1K tokens (input + output combined estimate)
    MODEL_COSTS = {
        'gpt-4o-mini': 0.000375,      # avg of 0.15 + 0.6 / 1M * 1000
        'gpt-4.1-nano': 0.00075,      # avg of 0.1 + 1.4 / 1M * 1000
        'o4-mini': 0.00275,           # avg of 1.1 + 4.4 / 1M * 1000
    }
    
    def __init__(self, query_fn, check_answer_fn):
        """Initialize evaluator.
        
        Args:
            query_fn: Function(model, problem) -> response_text
            check_answer_fn: Function(response, answer) -> bool
        """
        self.query_fn = query_fn
        self.check_answer_fn = check_answer_fn
        
    def evaluate_router(
        self,
        router,
        dataset: List[Problem],
        verbose: bool = False,
    ) -> RoutingMetrics:
        """Evaluate a router on a dataset.
        
        Args:
            router: Router instance to evaluate
            dataset: List of Problem instances
            verbose: Print per-problem results
            
        Returns:
            RoutingMetrics with performance summary
        """
        total_cost = 0.0
        correct = 0
        model_usage = {'gpt-4o-mini': 0, 'gpt-4.1-nano': 0, 'o4-mini': 0}
        
        for i, problem in enumerate(dataset):
            # Route to model
            context = {'difficulty': problem.difficulty}
            model = router.route(problem.question, context)
            model_usage[model] += 1
            
            # Query model
            response = self.query_fn(model, problem.question)
            
            # Check correctness
            is_correct = self.check_answer_fn(response, problem.answer)
            correct += int(is_correct)
            
            # Track cost (simplified - uses avg estimate)
            total_cost += self.MODEL_COSTS[model]
            
            if verbose:
                status = "✓" if is_correct else "✗"
                logger.info(
                    f"Problem {i+1}/{len(dataset)} [{status}]: "
                    f"{model} (cost: ${self.MODEL_COSTS[model]:.6f})"
                )
        
        accuracy = correct / len(dataset) if dataset else 0.0
        avg_cost = total_cost / len(dataset) if dataset else 0.0
        efficiency = accuracy / total_cost if total_cost > 0 else 0.0
        
        metrics = RoutingMetrics(
            accuracy=accuracy,
            total_cost=total_cost,
            avg_cost_per_problem=avg_cost,
            efficiency=efficiency,
            model_usage=model_usage,
            num_problems=len(dataset),
            correct_count=correct,
        )
        
        if verbose:
            logger.info(f"\nRouter Evaluation:")
            logger.info(f"  Accuracy: {accuracy:.2%}")
            logger.info(f"  Total Cost: ${total_cost:.6f}")
            logger.info(f"  Efficiency: {efficiency:.2f} acc/$")
            logger.info(f"  Model Usage: {model_usage}")
        
        return metrics
    
    def evaluate_baselines(
        self,
        dataset: List[Problem],
        verbose: bool = False,
    ) -> Dict[str, RoutingMetrics]:
        """Evaluate fixed single-model baselines.
        
        Args:
            dataset: List of problems
            verbose: Print results
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        from .router import Router
        
        baselines = {}
        for model in ['gpt-4o-mini', 'gpt-4.1-nano', 'o4-mini']:
            # Create fixed router for this model
            logic = f"model = '{model}'"
            router = Router(logic=logic, name=f"fixed_{model}")
            
            metrics = self.evaluate_router(router, dataset, verbose=False)
            baselines[model] = metrics
            
            if verbose:
                logger.info(f"\n{model} baseline:")
                logger.info(f"  Accuracy: {metrics.accuracy:.2%}")
                logger.info(f"  Cost: ${metrics.total_cost:.6f}")
                logger.info(f"  Efficiency: {metrics.efficiency:.2f}")
        
        return baselines