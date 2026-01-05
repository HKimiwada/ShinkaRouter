"""Efficiency-aware evaluation for ShinkaRouter.

This evaluator computes a combined score that balances accuracy with efficiency,
driving evolution toward the Pareto frontier of accuracy vs. LLM calls.

SCORE SCALE NOTE:
- Set SCORE_SCALE="normalized" for score in [0,1] range (default)
- Set SCORE_SCALE="baseline" to match adas_aime baseline (0-100 accuracy)
"""

import argparse
import json
import os
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
from pathlib import Path
from shinka.core import run_shinka_eval
from config import ROUTER_CONFIG, get_lambda_schedule

# Score scale configuration
# "normalized": score = accuracy/100 - lambda*calls (range ~0-1)
# "baseline": score = accuracy - lambda*calls*100 (range ~0-100, matches adas_aime)
SCORE_SCALE = os.environ.get("SHINKA_SCORE_SCALE", "baseline")


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
    
    # Get primitive info if available
    primitive_info = ""
    if "primitive_calls" in false_answer.columns:
        primitives = false_answer.iloc[0].get("primitive_calls", [])
        if primitives:
            primitive_info = f"\n# Primitives used: {', '.join(primitives)}"
    
    text_feedback = (
        f"# Example of an AIME problem that could not be answered correctly:\n\n"
        f"{false_answer.iloc[0]['problem']}\n\n"
        f"# The Agent's wrong full response:\n\n{false_answer.iloc[0]['response']}\n\n"
        f"# The Agent's submit answer:\n\n{false_answer.iloc[0]['llm_answer']}\n\n"
        f"# The ground truth problem answer:\n\n{false_answer.iloc[0]['true_answer']}\n\n"
        f"# Number of LLM calls used: {false_answer.iloc[0]['num_llm_calls']}"
        f"{primitive_info}"
    )
    
    return text_feedback


def analyze_primitive_usage(all_df) -> Dict[str, Any]:
    """
    Analyze which primitives were used across all runs.
    
    Returns dict with:
        - primitive_counts: Counter of primitive usage
        - primitive_diversity: Number of unique primitives used
        - avg_chain_length: Average number of primitives per problem
        - most_common: Most frequently used primitive
    """
    all_primitives = []
    chain_lengths = []
    
    for df in all_df:
        if "primitive_calls" not in df.columns:
            continue
        for calls in df["primitive_calls"]:
            if isinstance(calls, list):
                all_primitives.extend(calls)
                chain_lengths.append(len(calls))
    
    if not all_primitives:
        return {
            "primitive_counts": {},
            "primitive_diversity": 0,
            "avg_chain_length": 1.0,
            "most_common": "unknown",
        }
    
    counts = Counter(all_primitives)
    
    return {
        "primitive_counts": dict(counts),
        "primitive_diversity": len(counts),
        "avg_chain_length": float(np.mean(chain_lengths)) if chain_lengths else 1.0,
        "most_common": counts.most_common(1)[0][0] if counts else "unknown",
    }


def compute_efficiency_score(
    accuracy: float,
    avg_calls: float,
    generation: int = 0,
) -> Tuple[float, float]:
    """
    Compute the efficiency-aware combined score.
    
    Score formula depends on SCORE_SCALE:
    - "normalized": score = accuracy/100 - (lambda × avg_calls)
    - "baseline": score = accuracy - (lambda × avg_calls × 100)
    
    Args:
        accuracy: Accuracy percentage (0-100)
        avg_calls: Average LLM calls per problem
        generation: Current generation (for adaptive lambda)
        
    Returns:
        Tuple of (combined_score, lambda_used)
    """
    lambda_val = get_lambda_schedule(generation)
    
    if SCORE_SCALE == "normalized":
        # Normalized scale (0-1 range)
        accuracy_norm = accuracy / 100.0
        efficiency_penalty = lambda_val * avg_calls
        combined_score = accuracy_norm - efficiency_penalty
        
        # Minimum accuracy threshold (normalized)
        if accuracy < ROUTER_CONFIG.min_accuracy_threshold * 100:
            combined_score = -1.0
    else:
        # Baseline scale (0-100 range, matching adas_aime)
        # Scale lambda to work with 0-100 accuracy
        efficiency_penalty = lambda_val * avg_calls * 100
        combined_score = accuracy - efficiency_penalty
        
        # Minimum accuracy threshold
        if accuracy < ROUTER_CONFIG.min_accuracy_threshold * 100:
            combined_score = -100.0
    
    return combined_score, lambda_val


def aggregate_metrics_with_efficiency(
    results: List[Tuple[float, float, float, float, Any]],
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
            "combined_score": -100.0 if SCORE_SCALE == "baseline" else -1.0,
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
    if SCORE_SCALE == "normalized":
        efficiency_penalty = lambda_used * avg_calls
    else:
        efficiency_penalty = lambda_used * avg_calls * 100
    
    # Analyze primitive usage
    primitive_stats = analyze_primitive_usage(all_df)
    
    # Compute variance across runs
    calls_per_run = [calls / proc for calls, proc in zip(all_num_llm_calls, all_processed)]
    
    # Public metrics (visible to evolution)
    public_metrics = {
        "accuracy": accuracy,
        "avg_calls": avg_calls,
        "cost": avg_cost,
        "efficiency_penalty": efficiency_penalty,
        "lambda_used": lambda_used,
        "calls_std": float(np.std(calls_per_run)) if len(calls_per_run) > 1 else 0.0,
        "accuracy_std": float(np.std(all_performance)) if len(all_performance) > 1 else 0.0,
        "primitive_diversity": primitive_stats["primitive_diversity"],
        "avg_chain_length": primitive_stats["avg_chain_length"],
        "most_common_primitive": primitive_stats["most_common"],
    }
    
    # Private metrics (stored but not used for evolution)
    private_metrics = {
        "all_performance": all_performance,
        "all_cost": all_cost,
        "all_processed": all_processed,
        "all_num_llm_calls": all_num_llm_calls,
        "pareto_coords": (avg_calls, accuracy),
        "generation": generation,
        "score_scale": SCORE_SCALE,
        "primitive_counts": primitive_stats["primitive_counts"],
    }
    
    # Extra data stored as pickle
    extra_data = {
        "df": all_df,
        "primitive_stats": primitive_stats,
    }
    
    # Text feedback
    text_feedback = construct_text_feedback(all_df)
    text_feedback += f"\n\n# Efficiency Metrics:\n"
    text_feedback += f"# Average LLM calls: {avg_calls:.2f}\n"
    text_feedback += f"# Lambda: {lambda_used:.4f}\n"
    text_feedback += f"# Efficiency penalty: {efficiency_penalty:.4f}\n"
    text_feedback += f"# Combined score: {combined_score:.4f} (scale: {SCORE_SCALE})\n"
    text_feedback += f"\n# Primitive Usage:\n"
    for prim, count in primitive_stats["primitive_counts"].items():
        text_feedback += f"#   {prim}: {count}\n"
    
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
    generation: int = 0,
) -> None:
    """
    Run efficiency-aware evaluation for ShinkaRouter.
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
    print(f"Score scale: {SCORE_SCALE}")
    print(f"=" * 80)
    
    from functools import partial
    
    # Create kwargs provider
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
        print(f"  Primitive Diversity: {metrics['public']['primitive_diversity']}")
        print(f"  Most Common: {metrics['public']['most_common_primitive']}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Evaluation failed: {error}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShinkaRouter efficiency-aware evaluation"
    )
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--num_experiment_runs", type=int, default=3)
    parser.add_argument("--max_calls", type=int, default=10)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument(
        "--score_scale",
        type=str,
        choices=["normalized", "baseline"],
        default=None,
        help="Score scale: 'normalized' (0-1) or 'baseline' (0-100)",
    )
    
    args = parser.parse_args()
    
    # Override score scale if provided via CLI
    if args.score_scale:
        SCORE_SCALE = args.score_scale
    
    main(
        args.program_path,
        args.results_dir,
        args.model_name,
        args.year,
        args.num_experiment_runs,
        args.max_calls,
        args.generation,
    )