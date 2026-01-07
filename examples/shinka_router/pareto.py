"""
Pareto frontier analysis for cost vs. accuracy trade-offs.
"""
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from .evaluator import RoutingMetrics


def is_dominated(point1: Tuple[float, float], point2: Tuple[float, float]) -> bool:
    """Check if point1 is dominated by point2 (minimizing cost, maximizing accuracy).
    
    Args:
        point1: (cost, accuracy) tuple
        point2: (cost, accuracy) tuple
        
    Returns:
        True if point2 dominates point1
    """
    cost1, acc1 = point1
    cost2, acc2 = point2
    # point2 dominates if it has lower cost AND higher accuracy (or equal on both)
    return (cost2 <= cost1 and acc2 >= acc1) and (cost2 < cost1 or acc2 > acc1)


def compute_pareto_frontier(
    metrics_list: List[RoutingMetrics],
) -> List[int]:
    """Identify non-dominated solutions.
    
    Args:
        metrics_list: List of routing metrics
        
    Returns:
        Indices of solutions on Pareto frontier
    """
    points = [(m.total_cost, m.accuracy) for m in metrics_list]
    n = len(points)
    is_pareto = [True] * n
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i != j and is_dominated(points[i], points[j]):
                is_pareto[i] = False
                break
    
    return [i for i in range(n) if is_pareto[i]]


def compute_hypervolume(
    metrics_list: List[RoutingMetrics],
    ref_point: Tuple[float, float] = None,
) -> float:
    """Compute hypervolume indicator (area dominated by Pareto set).
    
    Args:
        metrics_list: List of metrics
        ref_point: Reference point (max_cost, min_accuracy). Auto-computed if None.
        
    Returns:
        Hypervolume value
    """
    if not metrics_list:
        return 0.0
    
    points = [(m.total_cost, m.accuracy) for m in metrics_list]
    
    if ref_point is None:
        max_cost = max(p[0] for p in points) * 1.1
        min_acc = min(p[1] for p in points) * 0.9
        ref_point = (max_cost, min_acc)
    
    # Sort by cost
    sorted_points = sorted(points, key=lambda p: p[0])
    
    # Compute area
    hv = 0.0
    prev_cost = 0.0
    for cost, acc in sorted_points:
        width = ref_point[0] - cost
        height = acc - ref_point[1]
        hv += width * height
    
    return hv


def plot_pareto_frontier(
    evolved_metrics: List[RoutingMetrics],
    baseline_metrics: dict,
    save_path: str = None,
    title: str = "Router Pareto Frontier: Cost vs. Accuracy",
):
    """Plot cost vs accuracy with Pareto frontier highlighted.
    
    Args:
        evolved_metrics: Metrics from evolved routers
        baseline_metrics: Dict mapping model names to baseline metrics
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot evolved routers
    evolved_costs = [m.total_cost for m in evolved_metrics]
    evolved_accs = [m.accuracy * 100 for m in evolved_metrics]
    ax.scatter(evolved_costs, evolved_accs, alpha=0.6, s=100, 
               label='Evolved Routers', color='blue')
    
    # Plot baselines
    baseline_costs = [m.total_cost for m in baseline_metrics.values()]
    baseline_accs = [m.accuracy * 100 for m in baseline_metrics.values()]
    ax.scatter(baseline_costs, baseline_accs, alpha=0.8, s=150,
               marker='s', label='Baselines', color='red')
    
    # Annotate baselines
    for name, metrics in baseline_metrics.items():
        ax.annotate(name, (metrics.total_cost, metrics.accuracy * 100),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Highlight Pareto frontier
    all_metrics = list(evolved_metrics) + list(baseline_metrics.values())
    pareto_indices = compute_pareto_frontier(all_metrics)
    
    pareto_costs = [all_metrics[i].total_cost for i in pareto_indices]
    pareto_accs = [all_metrics[i].accuracy * 100 for i in pareto_indices]
    
    # Sort for line
    sorted_pairs = sorted(zip(pareto_costs, pareto_accs))
    pareto_costs_sorted = [p[0] for p in sorted_pairs]
    pareto_accs_sorted = [p[1] for p in sorted_pairs]
    
    ax.plot(pareto_costs_sorted, pareto_accs_sorted, 'g--', 
            linewidth=2, label='Pareto Frontier', alpha=0.7)
    ax.scatter(pareto_costs, pareto_accs, s=200, facecolors='none',
               edgecolors='green', linewidths=2)
    
    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def compute_dominated_area_percentage(
    evolved_metrics: List[RoutingMetrics],
    baseline_metrics: dict,
) -> float:
    """Compute percentage of baseline convex hull dominated by evolved solutions.
    
    Args:
        evolved_metrics: Metrics from evolved routers
        baseline_metrics: Baseline metrics dict
        
    Returns:
        Percentage of dominated area [0-100]
    """
    # Simplified: compute how many baselines are dominated
    baselines = list(baseline_metrics.values())
    dominated_count = 0
    
    for baseline in baselines:
        baseline_point = (baseline.total_cost, baseline.accuracy)
        for evolved in evolved_metrics:
            evolved_point = (evolved.total_cost, evolved.accuracy)
            if is_dominated(baseline_point, evolved_point):
                dominated_count += 1
                break
    
    return (dominated_count / len(baselines)) * 100 if baselines else 0.0