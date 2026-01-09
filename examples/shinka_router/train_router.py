"""
Train ShinkaRouter on AIME 2024 problems.
Evolves routing logic across gpt-4o-mini, gpt-4.1-nano, o4-mini.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List
import pandas as pd
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shinka.llm.llm import LLMClient as LLM
from examples.shinka_router.router import Router
from examples.shinka_router.evaluator import RouterEvaluator, Problem, RoutingMetrics
from examples.shinka_router.evolution import RouterEvolution
from examples.shinka_router.pareto import (
    plot_pareto_frontier,
    compute_pareto_frontier,
    compute_dominated_area_percentage,
)
from examples.shinka_router.math_eval import evaluate_math_correctness

# Load environment
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_aime_problems(year: int = 2024, num_problems: int = None) -> List[Problem]:
    """Load AIME problems from CSV."""
    csv_path = "examples/shinka_router/AIME_Dataset_1983_2025.csv"
    if not os.path.exists(csv_path):
        csv_path = "AIME_Dataset_1983_2025.csv"
    
    df = pd.read_csv(csv_path)
    year_col = "Year" if "Year" in df.columns else "year"
    df = df[df[year_col] == year]
    
    if num_problems:
        df = df.head(num_problems)
    
    problems = []
    for _, row in df.iterrows():
        question = row.get('problem', row.get('Problem', ""))
        answer = str(row.get('answer', row.get('Answer', "")))
        problems.append(Problem(question=question, answer=answer, difficulty="hard"))
    
    return problems


def create_query_fn():
    """Create real query function for AIME problems."""
    model_clients = {
        'gpt-4o-mini': LLM(model_names='gpt-4o-mini', temperatures=0.0, verbose=False),
        'gpt-4.1-nano': LLM(model_names='gpt-4.1-nano', temperatures=0.0, verbose=False),
        'o4-mini': LLM(model_names='o4-mini', temperatures=1.0, verbose=False),
    }
    
    def query(model: str, problem: str) -> str:
        client = model_clients[model]
        result = client.query(
            msg=f"Solve this AIME problem: {problem}\nFinal answer must be in \\boxed{{}} format.",
            system_msg="You are a competition mathematician. Provide a clear solution and end with \\boxed{answer}."
        )
        return result.content if result else ""
    
    return query


def check_answer_fn(response: str, answer: str) -> bool:
    """Check if response contains correct answer."""
    _, _, correct = evaluate_math_correctness(response, answer)
    return correct


def run_training(
    dataset: List[Problem],
    num_generations: int = 10,
    population_size: int = 8,
    output_dir: Path = Path("./aime_router_results"),
):
    """Train ShinkaRouter on AIME problems."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize components
    query_fn = create_query_fn()
    evaluator = RouterEvaluator(query_fn=query_fn, check_answer_fn=check_answer_fn)
    
    # LLM for generating mutations
    mutation_llm = LLM(
        model_names=["gpt-4o-mini"],
        temperatures=0.7,
        max_tokens=2048,
        verbose=False,
    )
    
    system_prompt = """You are an expert at designing routing algorithms for multi-model AI systems solving competition math problems.

Available models:
- gpt-4o-mini: Cheapest ($0.000375/problem), handles simpler problems
- gpt-4.1-nano: Medium cost ($0.00075/problem), handles moderate complexity
- o4-mini: Most expensive ($0.00275/problem), best for complex reasoning

Your routing logic should analyze AIME problems and route to the most cost-effective model.
Consider: problem length, mathematical keywords, equation complexity, multiple steps, etc.

Output Python code that assigns variable 'model' to one of the three model names."""
    
    mutation_prompt = """Improve this router for AIME competition math problems:

# Current Router ({parent_name}):
```python
{parent_logic}
```

Create an improved version that better identifies which problems need expensive reasoning models.
Consider: proof keywords, equation density, multi-step indicators, specific math domains.

Output only Python code between ```python and ``` tags.
Add short name between <n> and </n> tags."""
    
    evolution = RouterEvolution(
        llm_client=mutation_llm,
        system_prompt=system_prompt,
        mutation_prompt=mutation_prompt,
    )
    
    # Evaluate baselines
    logger.info("Evaluating baseline models on AIME 2024...")
    baselines = evaluator.evaluate_baselines(dataset, verbose=True)
    
    # Initialize population with AIME-specific routers
    logger.info(f"Initializing population of {population_size} routers...")
    population = evolution.initialize_population(size=min(population_size, 4))
    
    # Add AIME-specific initial routers
    aime_routers = [
        Router("""
# Math keyword density
keywords = ['prove', 'show', 'find', 'determine', 'compute', 'integral', 'derivative']
kw_count = sum(1 for kw in keywords if kw in problem.lower())
if kw_count >= 3 or 'prove' in problem.lower():
    model = 'o4-mini'
elif kw_count >= 1:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""", "math_keywords"),
        Router("""
# Equation complexity
import re
equations = len(re.findall(r'[=<>]', problem))
numbers = len(re.findall(r'\\d+', problem))
if equations >= 3 or (equations >= 2 and numbers >= 5):
    model = 'o4-mini'
elif equations >= 1:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""", "equation_complex"),
    ]
    population.extend(aime_routers[:population_size - len(population)])
    
    # Evolution loop
    pop_metrics = [(r, evaluator.evaluate_router(r, dataset)) for r in population]
    all_routers, all_metrics = [], []
    generation_best = []
    
    logger.info(f"\nStarting evolution for {num_generations} generations...")
    
    for gen in range(num_generations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {gen + 1}/{num_generations}")
        logger.info(f"{'='*60}")
        
        pop_metrics.sort(key=lambda x: x[1].efficiency, reverse=True)
        
        best = pop_metrics[0]
        logger.info(f"Best: {best[0].name}")
        logger.info(f"  Accuracy: {best[1].accuracy:.2%} ({best[1].correct_count}/{best[1].num_problems})")
        logger.info(f"  Cost: ${best[1].total_cost:.6f}")
        logger.info(f"  Efficiency: {best[1].efficiency:.4f}")
        logger.info(f"  Models: {best[1].model_usage}")
        
        generation_best.append(best)
        all_routers.extend([r for r, _ in pop_metrics])
        all_metrics.extend([m for _, m in pop_metrics])
        
        # Selection
        num_parents = max(2, population_size // 2)
        parents = [r for r, _ in pop_metrics[:num_parents]]
        
        # Generate offspring
        offspring = []
        
        # Mutations
        num_mutations = int(population_size * 0.7)
        for i in range(num_mutations):
            parent = parents[i % len(parents)]
            inspirations = [p for p in parents if p != parent]
            mutant = evolution.mutate_router(parent, inspirations)
            if mutant:
                offspring.append(mutant)
        
        # Crossovers
        import random
        num_crossovers = population_size - len(offspring)
        for _ in range(num_crossovers):
            p1, p2 = random.sample(parents, 2)
            child = evolution.crossover(p1, p2)
            if child:
                offspring.append(child)
        
        # Evaluate offspring
        offspring_metrics = [(r, evaluator.evaluate_router(r, dataset)) for r in offspring]
        
        # Select next generation
        pop_metrics = pop_metrics[:num_parents] + offspring_metrics
        pop_metrics.sort(key=lambda x: x[1].efficiency, reverse=True)
        pop_metrics = pop_metrics[:population_size]
        population = [r for r, _ in pop_metrics]
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("Evolution Complete")
    logger.info(f"{'='*60}")
    
    pareto_indices = compute_pareto_frontier(all_metrics)
    pareto_routers = [all_routers[i] for i in pareto_indices]
    pareto_metrics = [all_metrics[i] for i in pareto_indices]
    
    logger.info(f"\nPareto Frontier: {len(pareto_routers)} routers")
    for r, m in zip(pareto_routers, pareto_metrics):
        logger.info(f"  {r.name}: {m.accuracy:.2%}, ${m.total_cost:.6f}, eff={m.efficiency:.4f}")
    
    dominated_pct = compute_dominated_area_percentage(all_metrics, baselines)
    logger.info(f"\nDominated baseline area: {dominated_pct:.1f}%")
    
    # Save results
    plot_pareto_frontier(all_metrics, baselines, save_path=str(output_dir / "pareto.png"))
    
    results = {
        'baselines': {k: v.to_dict() for k, v in baselines.items()},
        'pareto_frontier': [
            {'router': r.to_dict(), 'metrics': m.to_dict()}
            for r, m in zip(pareto_routers, pareto_metrics)
        ],
        'generation_best': [
            {'router': r.to_dict(), 'metrics': m.to_dict()}
            for r, m in generation_best
        ],
        'dominated_area_pct': dominated_pct,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best router code
    best_router = generation_best[-1][0]
    with open(output_dir / "best_router.py", 'w') as f:
        f.write(f"# Best Router: {best_router.name}\n\n{best_router.logic}")
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    # Configuration
    NUM_PROBLEMS = 30  # Use subset for faster training
    NUM_GENERATIONS = 10
    POPULATION_SIZE = 10
    
    # Load AIME 2024 problems
    problems = load_aime_problems(year=2024, num_problems=NUM_PROBLEMS)
    logger.info(f"Loaded {len(problems)} AIME 2024 problems")
    
    # Run training
    run_training(
        dataset=problems,
        num_generations=NUM_GENERATIONS,
        population_size=POPULATION_SIZE,
        output_dir=Path("./2024_30_AIME_router_results"),
    )