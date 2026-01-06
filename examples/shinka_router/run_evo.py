#!/usr/bin/env python3
"""
ShinkaRouter Evolution Runner

This script configures and launches the evolution of routing logic
for the AIME math agent, balancing accuracy with efficiency.
"""

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from config import ROUTER_CONFIG


# Task description for the LLM architect
ROUTER_TASK_MSG = """You are an expert AI researcher designing ROUTING LOGIC for a mathematical problem-solving agent.

# AGENT ARCHITECTURE

The agent has these PRIMITIVES available (DO NOT modify these):

0. baseline_solve(problem): Exact match to adas_aime baseline (temp=0.0, simple prompt)
   - This is the STARTING POINT - evolution should improve on this
   - Uses: temperature=0.0, system="You are a skilled mathematician."

1. quick_solve(problem): Fast solving with temperature=0.7, good for easy problems
   - Higher temperature = more randomness = worse for math usually
   - Consider using baseline_solve or deep_think instead for accuracy

2. deep_think(problem): Careful chain-of-thought reasoning with temperature=0.0
   - Adds "Think step-by-step" instruction
   - Often better than baseline_solve for complex problems

3. verify(problem, answer): Skeptical validator checking a proposed solution
   - Use after getting an initial answer to catch errors

4. python_calc(problem): Systematic step-by-step calculation approach
   - Good for computation-heavy problems

5. ensemble_vote(problem, n): Generate n solutions and vote on answer
   - Uses temperature=0.7 for diversity, then majority vote
   - Costs n LLM calls

6. self_critique(problem, draft): Generate solution then critique and refine
   - Pass the FULL draft response, not just the answer

7. estimate_difficulty(problem): Returns 'easy', 'medium', or 'hard'
   - Use for adaptive routing decisions

8. classify_problem_type(problem): Returns problem category
   - algebra, geometry, number_theory, combinatorics, calculus

Your task is to evolve the `forward(problem)` method to INTELLIGENTLY ROUTE between these primitives.

# OPTIMIZATION OBJECTIVE

**Fitness Function:** Score = Accuracy - (λ × AvgCalls × 100)

Where:
- Accuracy: Percentage of AIME problems solved correctly (0-100%)
- AvgCalls: Average number of LLM calls per problem
- λ: Efficiency penalty (starts at 0.0, gradually increases to 0.10)

**Goal:** Find routing strategies on the PARETO FRONTIER - maximize accuracy while minimizing LLM calls.

# IMPORTANT: Temperature and Accuracy

For math problems, **lower temperature = better accuracy**:
- temperature=0.0: Deterministic, best for math (baseline_solve, deep_think)
- temperature=0.7: Random, worse for math but adds diversity (quick_solve, ensemble_vote)

The baseline (baseline_solve) uses temperature=0.0. To beat it, you need SMARTER routing, not just different primitives.

# CONSTRAINTS

- Maximum 10 LLM calls per problem (hard limit, exceeding causes failure)
- Must return 3-digit answer format (0-999) in \\boxed{{}} 
- Minimum 30% accuracy required (below this is heavily penalized)
- Lambda increases over generations, gradually adding efficiency pressure

# ROUTING STRATEGIES TO EXPLORE

1. **Chain-of-thought enhancement:**
   - Use deep_think instead of baseline_solve
   - deep_think adds explicit CoT prompting which often helps

2. **Sequential verification:**
   - baseline_solve/deep_think → verify
   - Catches errors in initial solution

3. **Difficulty-based routing:**
   - estimate_difficulty() to classify problem
   - Route easy→baseline_solve (1 call), hard→deep_think+verify (2 calls)

4. **Self-refinement:**
   - deep_think → self_critique
   - Iteratively improve the solution

5. **Ensemble for hard problems:**
   - ensemble_vote(n=3) for medium-difficulty problems
   - More diverse solutions, majority vote

6. **Multi-stage pipelines:**
   - estimate_difficulty → route to appropriate depth
   - Track call budget, adapt strategy

# CURRENT GENERATION INFO

The lambda penalty starts at 0.0 and increases gradually:
- Gen 0-9: λ=0.0 (pure accuracy, ignore efficiency)
- Gen 10-19: λ=0.01 (slight efficiency pressure)
- Gen 20-29: λ=0.02
- Gen 30-39: λ=0.03
- Gen 40-49: λ=0.05
- Gen 50-59: λ=0.07
- Gen 60+: λ=0.10 (strong efficiency pressure)

Early generations should focus on maximizing accuracy. Later generations should optimize the accuracy-efficiency tradeoff.

# PERFORMANCE FEEDBACK

You will receive:
1. combined_score: The fitness value (accuracy - λ×calls×100)
2. accuracy: Raw accuracy percentage
3. avg_calls: Average LLM calls per problem
4. text_feedback: Examples of failed problems to learn from
5. lambda_used: Current λ value for this generation
6. primitive_diversity: How many different primitives were used

# CODE REQUIREMENTS

- Only modify the `forward(self, problem)` method inside EVOLVE-BLOCK
- DO NOT change primitive methods (baseline_solve, verify, etc.)
- Return (response, total_cost) tuple
- Handle all primitives returning (response, cost) tuples
- Track and sum costs across all primitive calls
- Call self.reset_tracking() at the start of forward()

Example improved forward():
```python
def forward(self, problem: str) -> Tuple[str, float]:
    self.reset_tracking()
    total_cost = 0.0
    
    # Use deep_think for better CoT reasoning
    response, cost = self.deep_think(problem)
    total_cost += cost
    
    # Verify the answer
    answer = self.extract_boxed_answer(response)
    if answer:
        verify_response, verify_cost = self.verify(problem, answer)
        total_cost += verify_cost
        return verify_response, total_cost
    
    return response, total_cost
```

Remember: The baseline uses temperature=0.0. To beat it, use better REASONING strategies, not just different temperatures!
"""


def create_job_config():
    """Create job configuration for local evaluation."""
    return LocalJobConfig(
        eval_program_path="evaluate.py",
        extra_cmd_args={
            "model_name": "gpt-4.1-nano",
            "year": ROUTER_CONFIG.test_year,
            "max_calls": ROUTER_CONFIG.max_calls_per_problem,
            "num_experiment_runs": ROUTER_CONFIG.num_experiment_runs,
        },
    )


def create_database_config():
    """Create database configuration for evolution."""
    return DatabaseConfig(
        db_path="router_evolution.sqlite",
        num_islands=ROUTER_CONFIG.num_islands,
        archive_size=ROUTER_CONFIG.archive_size,
        
        # Inspiration parameters
        elite_selection_ratio=0.3,
        num_archive_inspirations=5,
        num_top_k_inspirations=2,
        
        # Island migration
        migration_interval=5,
        migration_rate=0.15,
        island_elitism=True,
        enforce_island_separation=True,
        
        # Parent selection
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
    )


def create_evolution_config():
    """Create evolution configuration."""
    return EvolutionConfig(
        task_sys_msg=ROUTER_TASK_MSG,
        
        # Patch generation
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.5, 0.3, 0.2],
        
        # Evolution parameters
        num_generations=ROUTER_CONFIG.num_generations,
        max_parallel_jobs=ROUTER_CONFIG.max_parallel_jobs,
        max_patch_resamples=3,
        max_patch_attempts=5,
        
        # Execution
        job_type="local",
        language="python",
        
        # LLM configuration
        llm_models=ROUTER_CONFIG.llm_models,
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 1.0],
            max_tokens=16384,
        ),
        
        # Meta-learning
        meta_rec_interval=10,
        meta_llm_models=ROUTER_CONFIG.meta_llm_models,
        meta_llm_kwargs=dict(temperatures=[0.0]),
        meta_max_recommendations=5,
        
        # Novelty and diversity
        embedding_model="text-embedding-3-small",
        code_embed_sim_threshold=0.95,
        max_novelty_attempts=3,
        novelty_llm_models=ROUTER_CONFIG.meta_llm_models,
        novelty_llm_kwargs=dict(temperatures=[0.0]),
        
        # Initial program
        init_program_path="initial.py",
        
        # Text feedback
        use_text_feedback=True,
    )


def main():
    """Run the ShinkaRouter evolution experiment."""
    print("=" * 80)
    print("ShinkaRouter Evolution")
    print("=" * 80)
    print(f"Generations: {ROUTER_CONFIG.num_generations}")
    print(f"Islands: {ROUTER_CONFIG.num_islands}")
    print(f"Test Year: {ROUTER_CONFIG.test_year}")
    print(f"Holdout Year: {ROUTER_CONFIG.holdout_year}")
    print(f"Lambda Schedule: 0.0 → 0.10 (adaptive)")
    print(f"Max Calls: {ROUTER_CONFIG.max_calls_per_problem}")
    print("=" * 80)
    
    # Create configurations
    job_config = create_job_config()
    db_config = create_database_config()
    evo_config = create_evolution_config()
    
    # Create and run evolution
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    
    print("\nStarting evolution...")
    print("Monitor progress with: shinka_visualize --port 8888 --open\n")
    
    runner.run()
    
    print("\n" + "=" * 80)
    print("Evolution complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run analyze_router.py to visualize Pareto frontier")
    print("2. Test best solution on holdout year")
    print("3. Examine evolved routing logic in best program")
    print("=" * 80)


if __name__ == "__main__":
    main()