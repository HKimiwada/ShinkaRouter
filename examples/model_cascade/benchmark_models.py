import os
import pandas as pd
import json
from tqdm import tqdm
# Import the class exactly as defined in shinka/llm/llm.py
from shinka.llm.llm import LLMClient as LLM
from examples.shinka_router.math_eval import evaluate_math_correctness

def run_benchmark(models_to_test, year=2024, num_problems=None):
    """
    Benchmarks specific models on AIME problems for a given year.
    """
    csv_path = "examples/shinka_router/AIME_Dataset_1983_2025.csv"
    
    if not os.path.exists(csv_path):
        csv_path = "AIME_Dataset_1983_2025.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Could not find AIME_Dataset_1983_2025.csv")

    df = pd.read_csv(csv_path)
    # Check for both casing variants of "Year"
    year_col = "Year" if "Year" in df.columns else "year"
    df = df[df[year_col] == year]
    
    if num_problems:
        df = df.head(num_problems)
    
    results = {model_name: {"correct": 0, "total": 0, "costs": [], "details": []} 
               for model_name in models_to_test}

    print(f"Benchmarking {len(models_to_test)} models on {len(df)} problems (AIME {year})...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Solving Problems"):
        problem_text = row.get('problem', row.get('Problem', ""))
        ground_truth = str(row.get('answer', row.get('Answer', "")))
        
        for model_name in models_to_test:
            # Initialize client with the specific model and temperature
            client = LLM(model_names=model_name, temperatures=0.0)
            
            try:
                # FIX: LLMClient.query uses 'msg' and 'system_msg'
                # Temperature is handled in __init__ or via llm_kwargs dict
                query_result = client.query(
                    msg=f"Solve this AIME problem: {problem_text}\nFinal answer must be in \\boxed{{}} format.",
                    system_msg="You are a competition mathematician. Provide a clear solution and end with \\boxed{answer}."
                )
                
                if query_result is None:
                    print(f"Query returned None for {model_name}")
                    continue

                # QueryResult object has .content and .cost attributes
                response = query_result.content
                cost = query_result.cost
                
                # Evaluate correctness
                extracted_ans, _, correct = evaluate_math_correctness(response, ground_truth)
                
                # Log results
                results[model_name]["total"] += 1
                if correct:
                    results[model_name]["correct"] += 1
                results[model_name]["costs"].append(cost)
                results[model_name]["details"].append({
                    "id": row.get('ID', index),
                    "correct": correct,
                    "extracted": extracted_ans,
                    "cost": cost
                })
            except Exception as e:
                print(f"\nError querying {model_name}: {e}")

    # Output Summary
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Avg Cost':<12}")
    print("-"*60)
    summary = []
    for model, data in results.items():
        total = data["total"]
        acc = (data["correct"] / total) * 100 if total > 0 else 0
        avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0
        print(f"{model:<20} | {acc:>8.2f}% | ${avg_cost:>10.6f}")
        summary.append({"model": model, "accuracy": acc, "avg_cost": avg_cost})
    print("="*60)
    
    return results, summary

if __name__ == "__main__":
    MODELS_LIST = ["gpt-4o-mini", "gpt-4.1-nano", "o4-mini-2025-04-16"]
    
    full_results, summary_stats = run_benchmark(MODELS_LIST, year=2024)
    
    output_file = "model_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=4)
    print(f"\nResults saved to {output_file}")