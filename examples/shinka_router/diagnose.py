#!/usr/bin/env python3
"""
ShinkaRouter Diagnostic Script

This script tests whether the primitives actually improve over the baseline.
If they don't, evolution has no room to improve.

Run this BEFORE evolution to understand the performance landscape.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_single_primitive(
    Agent,
    query_llm,
    primitive_name: str,
    problems: List[Dict],
    max_problems: int = 10,
) -> Tuple[float, float, List[bool]]:
    """
    Test a single primitive on a subset of problems.
    
    Returns: (accuracy, avg_cost, list of correct/incorrect)
    """
    from utils import create_call_limited_query_llm
    
    correct_list = []
    total_cost = 0.0
    
    for i, prob in enumerate(problems[:max_problems]):
        # Create fresh agent and query_llm for each problem
        limited_llm = create_call_limited_query_llm(query_llm, max_calls=10)
        agent = Agent(limited_llm)
        
        problem_text = prob['problem']
        true_answer = str(prob['answer']).strip()
        
        try:
            # Call the specific primitive
            if primitive_name == "baseline_solve":
                response, cost = agent.baseline_solve(problem_text)
            elif primitive_name == "deep_think":
                response, cost = agent.deep_think(problem_text)
            elif primitive_name == "quick_solve":
                response, cost = agent.quick_solve(problem_text)
            elif primitive_name == "python_calc":
                response, cost = agent.python_calc(problem_text)
            elif primitive_name == "ensemble_vote_3":
                response, cost = agent.ensemble_vote(problem_text, n=3)
            elif primitive_name == "deep_think_verify":
                response, cost = agent.deep_think(problem_text)
                answer = agent.extract_boxed_answer(response)
                if answer:
                    response, cost2 = agent.verify(problem_text, answer)
                    cost += cost2
            elif primitive_name == "deep_think_critique":
                response, cost = agent.deep_think(problem_text)
                response2, cost2 = agent.self_critique(problem_text, response)
                response = response2
                cost += cost2
            else:
                raise ValueError(f"Unknown primitive: {primitive_name}")
            
            # Extract answer
            llm_answer = agent.extract_boxed_answer(response)
            if llm_answer:
                llm_answer = llm_answer.strip().lstrip("0") or "0"
            else:
                llm_answer = ""
            
            true_answer_clean = true_answer.strip().lstrip("0") or "0"
            correct = (llm_answer == true_answer_clean)
            
            correct_list.append(correct)
            total_cost += cost
            
            status = "✓" if correct else "✗"
            print(f"  [{i+1}/{max_problems}] {status} LLM={llm_answer}, True={true_answer_clean}")
            
        except Exception as e:
            print(f"  [{i+1}/{max_problems}] ERROR: {e}")
            correct_list.append(False)
    
    accuracy = 100 * sum(correct_list) / len(correct_list) if correct_list else 0
    avg_cost = total_cost / len(correct_list) if correct_list else 0
    
    return accuracy, avg_cost, correct_list


def run_primitive_comparison(
    model_name: str = "gpt-4o-mini",
    year: int = 2024,
    max_problems: int = 10,
):
    """Compare all primitives on same problem set."""
    
    from utils import query_llm
    from initial import Agent
    
    # Load dataset
    dataset_path = Path(__file__).parent / "AIME_Dataset_1983_2025.csv"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    df = df[df["Year"] == year]
    
    if len(df) == 0:
        print(f"ERROR: No problems found for year {year}")
        return
    
    # Sample problems (same for all primitives)
    problems = df.sample(n=min(max_problems, len(df)), random_state=42).to_dict('records')
    
    print("=" * 80)
    print(f"ShinkaRouter Primitive Comparison")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Year: {year}")
    print(f"Problems: {len(problems)}")
    print("=" * 80)
    
    # Create base query function
    base_query = partial(query_llm, model_name=model_name)
    
    # Primitives to test
    primitives = [
        ("baseline_solve", "Baseline (temp=0.0, simple prompt)"),
        ("deep_think", "Deep Think (temp=0.0, CoT prompt)"),
        ("quick_solve", "Quick Solve (temp=0.7, simple prompt)"),
        ("python_calc", "Python Calc (temp=0.0, calculation prompt)"),
        ("ensemble_vote_3", "Ensemble Vote n=3 (temp=0.7, majority)"),
        ("deep_think_verify", "Deep Think + Verify (2 calls)"),
        ("deep_think_critique", "Deep Think + Self-Critique (2 calls)"),
    ]
    
    results = []
    
    for primitive_name, description in primitives:
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print("=" * 80)
        
        accuracy, avg_cost, correct_list = test_single_primitive(
            Agent, base_query, primitive_name, problems, max_problems
        )
        
        results.append({
            "primitive": primitive_name,
            "description": description,
            "accuracy": accuracy,
            "avg_cost": avg_cost,
            "correct_count": sum(correct_list),
            "total": len(correct_list),
        })
        
        print(f"\nResult: {accuracy:.1f}% accuracy, ${avg_cost:.4f} avg cost")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Primitive Comparison")
    print("=" * 80)
    print(f"{'Primitive':<25} {'Accuracy':>10} {'Cost':>10} {'Calls':>8}")
    print("-" * 60)
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    baseline_acc = None
    for r in results:
        if r['primitive'] == 'baseline_solve':
            baseline_acc = r['accuracy']
        
        calls = "1" if "vote" not in r['primitive'] and "verify" not in r['primitive'] and "critique" not in r['primitive'] else "2-3"
        print(f"{r['primitive']:<25} {r['accuracy']:>9.1f}% ${r['avg_cost']:>9.4f} {calls:>8}")
    
    print("-" * 60)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if baseline_acc is not None:
        improvements = [(r['primitive'], r['accuracy'] - baseline_acc) 
                       for r in results if r['primitive'] != 'baseline_solve']
        
        better = [p for p, delta in improvements if delta > 0]
        same = [p for p, delta in improvements if delta == 0]
        worse = [p for p, delta in improvements if delta < 0]
        
        if better:
            print(f"\n✓ BETTER than baseline: {', '.join(better)}")
            print("  → Evolution CAN improve by using these primitives")
        
        if same:
            print(f"\n= SAME as baseline: {', '.join(same)}")
            print("  → These primitives don't help")
        
        if worse:
            print(f"\n✗ WORSE than baseline: {', '.join(worse)}")
            print("  → Avoid these primitives")
        
        if not better:
            print("\n⚠️  WARNING: No primitive beats baseline!")
            print("   This explains why evolution stagnates.")
            print("   Possible causes:")
            print("   1. Model is at its ceiling for AIME")
            print("   2. Prompting variations don't help this model")
            print("   3. Need a stronger model (gpt-4o, o1-mini)")
    
    print("=" * 80)
    
    return results


def test_model_baseline(
    models: List[str] = None,
    year: int = 2024,
    max_problems: int = 10,
):
    """Test baseline_solve across different models to find ceiling."""
    
    if models is None:
        models = ["gpt-4o-mini"]  # Add more if you have access
    
    from utils import query_llm
    from initial import Agent
    
    # Load dataset
    df = pd.read_csv("AIME_Dataset_1983_2025.csv")
    df = df[df["Year"] == year]
    problems = df.sample(n=min(max_problems, len(df)), random_state=42).to_dict('records')
    
    print("=" * 80)
    print("Model Comparison (baseline_solve)")
    print("=" * 80)
    
    for model in models:
        print(f"\nTesting model: {model}")
        base_query = partial(query_llm, model_name=model)
        
        accuracy, avg_cost, _ = test_single_primitive(
            Agent, base_query, "baseline_solve", problems, max_problems
        )
        
        print(f"  → {model}: {accuracy:.1f}% accuracy, ${avg_cost:.4f} cost")
    
    print("=" * 80)


def check_determinism(
    model_name: str = "gpt-4o-mini",
    year: int = 2024,
    num_runs: int = 3,
):
    """Check if temp=0.0 gives deterministic results."""
    
    from utils import query_llm, create_call_limited_query_llm
    from initial import Agent
    
    df = pd.read_csv("AIME_Dataset_1983_2025.csv")
    df = df[df["Year"] == year]
    problem = df.iloc[0]  # First problem
    
    print("=" * 80)
    print("Determinism Check (temp=0.0)")
    print("=" * 80)
    print(f"Problem: {problem['problem'][:100]}...")
    print(f"True answer: {problem['answer']}")
    print()
    
    base_query = partial(query_llm, model_name=model_name)
    answers = []
    
    for i in range(num_runs):
        limited_llm = create_call_limited_query_llm(base_query, max_calls=10)
        agent = Agent(limited_llm)
        
        response, cost = agent.baseline_solve(problem['problem'])
        answer = agent.extract_boxed_answer(response)
        answers.append(answer)
        print(f"  Run {i+1}: {answer}")
    
    if len(set(answers)) == 1:
        print(f"\n✓ Deterministic: All {num_runs} runs gave same answer")
        print("  → This means accuracy won't change between runs")
        print("  → Evolution sees no variance to optimize")
    else:
        print(f"\n⚠ Non-deterministic: Got {len(set(answers))} different answers")
        print(f"  → Answers: {set(answers)}")
    
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ShinkaRouter Diagnostics")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--test", type=str, default="primitives",
                       choices=["primitives", "models", "determinism", "all"])
    
    args = parser.parse_args()
    
    if args.test in ["primitives", "all"]:
        run_primitive_comparison(args.model, args.year, args.problems)
    
    if args.test in ["determinism", "all"]:
        check_determinism(args.model, args.year)
    
    if args.test in ["models", "all"]:
        test_model_baseline(year=args.year, max_problems=args.problems)


if __name__ == "__main__":
    main()