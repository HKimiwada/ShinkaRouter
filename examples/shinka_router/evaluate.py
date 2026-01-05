"""Efficiency-aware evaluation for ShinkaRouter.

This evaluator computes a combined score that balances accuracy with efficiency,
driving evolution toward the Pareto frontier of accuracy vs. LLM calls.
"""

import argparse
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path
from shinka.core import run_shinka_eval
from config import ROUTER_CONFIG, get_lambda_schedule


def construct_text_feedback(all_df) -> str:
    """Collect feedback from all wrong answers."""
    extra_dfs = [df.sort_values("id").reset_index(drop=True) for df in all_df]
    
    # Find ids where all runs had "correct" == False
    ids_all_incorrect = set.intersection(
        *[set(df.loc[df["correct"] == False, "id"]) for df in extra_dfs]
    )
    
    if not ids_all_incorrect:
        return "All problems were solved correctly in at least one run!"
    
    ids_all_incorrect = sorted(ids_all_incorrect)
    
    # Select from first dataframe
    df0_selected = extra_dfs[0][extra_dfs[0]["id"].isin(ids_all_incorrect)]
    random_id = df0_selected.sample(1)["id"].values[0]
    false_answer = df0_selected[df0_selected["id"] == random_id]
    
    text_feedback = (
        f"# Example of an AIME problem that could not be answered correctly:\n\n"
        f"{false_answer.iloc[0]['problem']}\n\n"
        f"# The Agent's wrong full response:\n\n{false_answer.iloc[0]['response']}\n\n"
        f"# The Agent's submit answer:\n\n{false_answer.iloc[0]['llm_answer']}\n\n"
        f"# The ground truth problem answer:\n\n{false_answer.iloc[0]['true_answer']}\n\n"
        f"# Number of LLM calls used: {false_answer.iloc[0]['num_llm_calls']}"
    )
    
    return text_feedback


def analyze_primitive_usage(all_df) -> Dict[str, float]:
    """
    Analyze which primitives were used (if tracked).
    This would require instrumenting the Agent class to log calls.
    """
    # Placeholder for future primitive usage tracking
    return {
        "primitive_diversity": 1.0,  # Could track unique primitives used
        "avg_depth": 1.0,  # Could track call chain depth
    }


def compute_efficiency_score(
    accuracy: float,
    avg_calls: float,
    generation: int = 0,
) -> Tuple[float, float]:
    """
    Compute the efficiency-aware combined score.
    
    Score = Accuracy - (lambda × AvgCalls)
    
    Args:
        accuracy: Accuracy percentage (0-100)
        avg_calls: Average LLM calls per problem
        generation: Current generation (for adaptive lambda)
        
    Returns:
        Tuple of (combined_score, lambda_used)
    """
    # Get adaptive lambda value
    lambda_val = get_lambda_schedule(generation)
    
    # Normalize accuracy to 0-1 scale
    accuracy_norm = accuracy / 100.0
    
    # Compute efficiency penalty
    efficiency_penalty = lambda_val * avg_calls
    
    # Combined score (can go negative if very inefficient)
    combined_score = accuracy_norm - efficiency_penalty
    
    # Apply minimum accuracy threshold
    if accuracy < ROUTER_CONFIG.min_accuracy_threshold * 100:
        combined_score = -999.0  # Heavily penalize low accuracy
    
    return combined_score, lambda_val


def aggregate_metrics_with_efficiency(
    results: List[Tuple[float, float, float, float]],
    generation: int = 0,
) -> Dict[str, Any]:
    """
    Aggregate results with efficiency-aware scoring.
    
    Args:
        results: List of (accuracy, cost, processed, num_llm_calls, df) tuples
        generation: Current generation number
        
    Returns:
        Dictionary containing public/private metrics and combined_score
    """
    if not results:
        return {
            "public": {
                "accuracy": 0.0,
                "avg_calls": 0.0,
                "cost": 0.0,
                "efficiency_penalty": 0.0,
                "lambda_used": 0.0,
            },
            "private": {"processed": 0},
            "combined_score": -999.0,
            "text_feedback": "No results generated.",
        }
    
    # Unpack results
    (
        all_performance,
        all_cost,
        all_processed,
        all_num_llm_calls,
        all_df,
    ) = zip(*results)
    
    # Compute aggregated metrics
    accuracy = float(np.mean(all_performance))
    total_processed = sum(all_processed)
    total_llm_calls = sum(all_num_llm_calls)
    avg_calls = float(total_llm_calls / total_processed) if total_processed > 0 else 0.0
    avg_cost = float(np.mean(all_cost))
    
    # Compute efficiency score
    combined_score, lambda_used = compute_efficiency_score(
        accuracy, avg_calls, generation
    )
    
    # Compute efficiency penalty for visibility
    efficiency_penalty = lambda_used * avg_calls
    
    # Analyze primitive usage (if available)
    primitive_stats = analyze_primitive_usage(all_df)
    
    # Public metrics (visible to evolution)
    public_metrics = {
        "accuracy": accuracy,
        "avg_calls": avg_calls,
        "cost": avg_cost,
        "efficiency_penalty": efficiency_penalty,
        "lambda_used": lambda_used,
        "calls_std": float(np.std([calls / proc for calls, proc in zip(all_num_llm_calls, all_processed)])),
        **primitive_stats,
    }
    
    # Private metrics (stored but not used for evolution)
    private_metrics = {
        "all_performance": all_performance,
        "all_cost": all_cost,
        "all_processed": all_processed,
        "all_num_llm_calls": all_num_llm_calls,
        "pareto_coords": (avg_calls, accuracy),  # For Pareto frontier plotting
        "generation": generation,
    }
    
    # Extra data stored as pickle
    extra_data = {
        "df": all_df,
        "primitive_usage": primitive_stats,
    }
    
    # Text feedback
    text_feedback = construct_text_feedback(all_df)
    text_feedback += f"\n\n# Efficiency Metrics:\n"
    text_feedback += f"# Average LLM calls: {avg_calls:.2f}\n"
    text_feedback += f"# Lambda: {lambda_used:.4f}\n"
    text_feedback += f"# Efficiency penalty: {efficiency_penalty:.4f}\n"
    text_feedback += f"# Combined score: {combined_score:.4f}"
    
    metrics = {
        "public": public_metrics,
        "private": private_metrics,
        "combined_score": float(combined_score),
        "extra_data": extra_data,
        "text_feedback": text_feedback,
    }
    
    return metrics


def get_experiment_kwargs(
    run_idx: int,
    model_name: str,
    year: int,
    max_calls: int,
) -> Dict[str, Any]:
    """Provides keyword arguments for each experiment run."""
    return {
        "model_name": model_name,
        "year": year,
        "max_calls": max_calls,
    }


def main(
    program_path: str,
    results_dir: str,
    model_name: str = "gpt-4o-mini",
    year: int = 2024,
    num_experiment_runs: int = 3,
    max_calls: int = 10,
    generation: int = 0,  # Added for adaptive lambda
) -> None:
    """
    Run efficiency-aware evaluation for ShinkaRouter.
    
    Args:
        program_path: Path to the program to evaluate
        results_dir: Directory to save results
        model_name: LLM model name
        year: AIME dataset year
        num_experiment_runs: Number of evaluation runs
        max_calls: Maximum LLM calls per problem
        generation: Current generation (for adaptive lambda)
    """
    print(f"=" * 80)
    print(f"ShinkaRouter Evaluation")
    print(f"=" * 80)
    print(f"Program: {program_path}")
    print(f"Results: {results_dir}")
    print(f"Model: {model_name}")
    print(f"Year: {year}")
    print(f"Max calls: {max_calls}")
    print(f"Runs: {num_experiment_runs}")
    print(f"Generation: {generation}")
    print(f"Lambda: {get_lambda_schedule(generation):.4f}")
    print(f"=" * 80)
    
    from functools import partial
    
    # Create kwargs provider with generation info
    get_kwargs_for_run = partial(
        get_experiment_kwargs,
        model_name=model_name,
        year=year,
        max_calls=max_calls,
    )
    
    # Create aggregator with generation info
    aggregate_fn = partial(
        aggregate_metrics_with_efficiency,
        generation=generation,
    )
    
    # Run evaluation
    metrics, correct, error = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_kwargs_for_run,
        aggregate_metrics_fn=aggregate_fn,
    )
    
    # Print results
    if correct:
        print("\n" + "=" * 80)
        print("Evaluation completed successfully!")
        print("=" * 80)
        print("\nMetrics:")
        print(f"  Accuracy: {metrics['public']['accuracy']:.2f}%")
        print(f"  Avg Calls: {metrics['public']['avg_calls']:.2f}")
        print(f"  Cost: ${metrics['public']['cost']:.4f}")
        print(f"  Lambda: {metrics['public']['lambda_used']:.4f}")
        print(f"  Efficiency Penalty: {metrics['public']['efficiency_penalty']:.4f}")
        print(f"  Combined Score: {metrics['combined_score']:.4f}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Evaluation failed: {error}")
        print("=" * 80)
        print("\nDefault metrics stored due to error:")
        for key, value in metrics.items():
            if key != 'text_feedback':
                print(f"  {key}: {value}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShinkaRouter efficiency-aware evaluation"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="AIME dataset year",
    )
    parser.add_argument(
        "--num_experiment_runs",
        type=int,
        default=3,
        help="Number of evaluation runs",
    )
    parser.add_argument(
        "--max_calls",
        type=int,
        default=10,
        help="Maximum LLM calls per problem",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=0,
        help="Current generation (for adaptive lambda)",
    )
    
    args = parser.parse_args()
    
    main(
        args.program_path,
        args.results_dir,
        args.model_name,
        args.year,
        args.num_experiment_runs,
        args.max_calls,
        args.generation,
    )