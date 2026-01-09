# Transfer found routing logic to 2023/2025 (as transfer logic found through 2024 may not be valid for other years)
"""
Validation script for evolved routers vs baseline models on AIME problems.
Combines benchmark_models.py functionality with evolved router evaluation.
"""
import os
import sys
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# --- Path Configuration ---
# Anchors all relative paths to the location of this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Add parent to path for imports
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from shinka.llm.llm import LLMClient as LLM
from examples.shinka_router.math_eval import evaluate_math_correctness
from examples.shinka_router.router import Router

def load_evolved_routers(results_dir: Path):
    """Load evolved routers from results JSON using an absolute Path object."""
    results_path = results_dir / "results.json"
    
    if not results_path.exists():
        print(f"Warning: Results file not found at {results_path}")
        return []
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    routers = []
    # Load Pareto frontier routers
    for entry in results.get('pareto_frontier', []):
        router_data = entry['router']
        router = Router.from_dict(router_data)
        metrics = entry['metrics']
        routers.append({
            'router': router,
            'name': router.name,
            'train_accuracy': metrics['accuracy'],
            'train_cost': metrics['total_cost'],
            'train_efficiency': metrics['efficiency']
        })
    
    # Load best router from final generation
    gen_best = results.get('generation_best', [])
    if gen_best:
        final_best = gen_best[-1]
        router_data = final_best['router']
        router = Router.from_dict(router_data)
        metrics = final_best['metrics']
        
        if not any(r['name'] == router.name for r in routers):
            routers.append({
                'router': router,
                'name': f"{router.name}_gen_best",
                'train_accuracy': metrics['accuracy'],
                'train_cost': metrics['total_cost'],
                'train_efficiency': metrics['efficiency']
            })
    
    return routers


def run_benchmark(models_to_test, csv_path: Path, year=2025, num_problems=None):
    """Benchmarks fixed models on AIME problems."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find AIME dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    year_col = "Year" if "Year" in df.columns else "year"
    df = df[df[year_col] == year]
    
    if num_problems:
        df = df.head(num_problems)
    
    results = {model_name: {"correct": 0, "total": 0, "costs": [], "details": []} 
               for model_name in models_to_test}

    print(f"\n{'='*70}")
    print(f"Benchmarking {len(models_to_test)} fixed models on {len(df)} problems (AIME {year})")
    print(f"{'='*70}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Solving Problems"):
        problem_text = row.get('problem', row.get('Problem', ""))
        ground_truth = str(row.get('answer', row.get('Answer', "")))
        
        for model_name in models_to_test:
            temp = 1.0 if ("o4" in model_name or "o1" in model_name) else 0.0
            client = LLM(model_names=model_name, temperatures=temp, verbose=False)
            
            try:
                query_result = client.query(
                    msg=f"Solve this AIME problem: {problem_text}\nFinal answer must be in \\boxed{{}} format.",
                    system_msg="You are a competition mathematician. Provide a clear solution and end with \\boxed{answer}."
                )
                
                if query_result:
                    extracted_ans, _, correct = evaluate_math_correctness(query_result.content, ground_truth)
                    results[model_name]["total"] += 1
                    if correct: results[model_name]["correct"] += 1
                    results[model_name]["costs"].append(query_result.cost)
                    results[model_name]["details"].append({
                        "id": row.get('ID', index),
                        "correct": correct,
                        "extracted": extracted_ans,
                        "cost": query_result.cost
                    })
            except Exception as e:
                print(f"\nError querying {model_name}: {e}")

    # Process and print summary
    summary = []
    for model, data in results.items():
        total = data["total"]
        acc = (data["correct"] / total) * 100 if total > 0 else 0
        avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0
        efficiency = (acc / 100) / avg_cost if avg_cost > 0 else 0
        summary.append({
            "model": model, "accuracy": acc, "avg_cost": avg_cost,
            "total_cost": sum(data["costs"]), "efficiency": efficiency,
            "correct": data["correct"], "total": total
        })
    
    return results, summary


def run_router_benchmark(routers, csv_path: Path, year=2025, num_problems=None):
    """Benchmarks evolved routers on AIME problems."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find AIME dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    year_col = "Year" if "Year" in df.columns else "year"
    df = df[df[year_col] == year]
    
    if num_problems:
        df = df.head(num_problems)
    
    model_clients = {
        'gpt-4o-mini': LLM(model_names='gpt-4o-mini', temperatures=0.0, verbose=False),
        'gpt-4.1-nano': LLM(model_names='gpt-4.1-nano', temperatures=0.0, verbose=False),
        'o4-mini': LLM(model_names='o4-mini', temperatures=1.0, verbose=False),
    }
    
    results = {r['name']: {
        "correct": 0, "total": 0, "costs": [], "details": [],
        "model_usage": {m: 0 for m in model_clients.keys()},
        "train_metrics": {'accuracy': r['train_accuracy'], 'cost': r['train_cost'], 'efficiency': r['train_efficiency']}
    } for r in routers}

    print(f"\n{'='*70}")
    print(f"Benchmarking {len(routers)} evolved routers on {len(df)} problems (AIME {year})")
    print(f"{'='*70}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Routing Problems"):
        problem_text = row.get('problem', row.get('Problem', ""))
        ground_truth = str(row.get('answer', row.get('Answer', "")))
        
        for router_info in routers:
            router, router_name = router_info['router'], router_info['name']
            try:
                selected_model = router.route(problem_text, {'difficulty': 'hard'})
                results[router_name]["model_usage"][selected_model] += 1
                
                client = model_clients[selected_model]
                query_result = client.query(
                    msg=f"Solve this AIME problem: {problem_text}\nFinal answer must be in \\boxed{{}} format.",
                    system_msg="You are a competition mathematician. Provide a clear solution and end with \\boxed{answer}."
                )
                
                if query_result:
                    extracted_ans, _, correct = evaluate_math_correctness(query_result.content, ground_truth)
                    results[router_name]["total"] += 1
                    if correct: results[router_name]["correct"] += 1
                    results[router_name]["costs"].append(query_result.cost)
                    results[router_name]["details"].append({
                        "id": row.get('ID', index), "correct": correct, "extracted": extracted_ans,
                        "cost": query_result.cost, "model_used": selected_model
                    })
            except Exception as e:
                print(f"\nError with router {router_name}: {e}")

    summary = []
    for r_name, data in results.items():
        total = data["total"]
        acc = (data["correct"] / total) * 100 if total > 0 else 0
        avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0
        efficiency = (acc / 100) / avg_cost if avg_cost > 0 else 0
        summary.append({
            "router": r_name, "accuracy": acc, "avg_cost": avg_cost,
            "total_cost": sum(data["costs"]), "efficiency": efficiency,
            "correct": data["correct"], "total": total, "model_usage": data["model_usage"],
            "train_accuracy": data['train_metrics']['accuracy'] * 100,
            "train_cost": data['train_metrics']['cost']
        })
    
    return results, summary


def compare_results(baseline_summary, router_summary):
    """Generate comparison between baselines and routers."""
    print("\n" + "="*70)
    print("COMPARISON: Baselines vs Evolved Routers")
    print("="*70)
    
    best_baseline = max(baseline_summary, key=lambda x: x['efficiency'])
    best_router = max(router_summary, key=lambda x: x['efficiency'])
    
    for label, best in [("Baseline", best_baseline), ("Router", best_router)]:
        name = best.get('model', best.get('router'))
        print(f"\nBest {label}: {name}")
        print(f"  Accuracy: {best['accuracy']:.2f}% | Efficiency: {best['efficiency']:.2f}")

    eff_diff = ((best_router['efficiency'] - best_baseline['efficiency']) / best_baseline['efficiency']) * 100
    print(f"\nImprovement in Efficiency: {eff_diff:+.2f}%")
    print("="*70)


def main():
    # --- Configuration ---
    YEAR = 2025
    NUM_PROBLEMS = 15
    BASELINE_MODELS = ["gpt-4o-mini", "gpt-4.1-nano", "o4-mini"]
    
    # Path Definitions (All relative to script folder)
    CSV_PATH = SCRIPT_DIR / "AIME_Dataset_1983_2025.csv"
    RESULTS_DIR = SCRIPT_DIR / "2024_15_AIME_router_results"
    OUTPUT_FILE = SCRIPT_DIR / "validation_results.json"
    
    # 1. Baseline Benchmark
    baseline_results, baseline_summary = run_benchmark(BASELINE_MODELS, CSV_PATH, YEAR, NUM_PROBLEMS)
    
    # 2. Router Loading
    routers = load_evolved_routers(RESULTS_DIR)
    if not routers:
        print("\nNo evolved routers found. Exiting.")
        return
    
    # 3. Router Benchmark
    router_results, router_summary = run_router_benchmark(routers, CSV_PATH, YEAR, NUM_PROBLEMS)
    
    # 4. Final Comparison
    compare_results(baseline_summary, router_summary)
    
    # 5. Save Data
    output = {
        "dataset": {"year": YEAR, "num_problems": NUM_PROBLEMS},
        "baselines": {"summary": baseline_summary},
        "routers": {"summary": router_summary}
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nValidation results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()