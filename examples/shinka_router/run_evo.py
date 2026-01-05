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
1. quick_solve(problem): Fast solving with temperature=0.7, good for easy problems
2. deep_think(problem): Careful chain-of-thought reasoning with temperature=0.0
3. verify(problem, answer): Skeptical validator checking a proposed solution
4. python_calc(problem): Solve using Python code for computational problems
5. ensemble_vote(problem, n): Generate n solutions and vote on answer
6. self_critique(problem, draft): Generate solution then critique and refine
7. estimate_difficulty(problem): Returns 'easy', 'medium', or 'hard'

Your task is to evolve the `forward(problem)` method to INTELLIGENTLY ROUTE between these primitives.

# OPTIMIZATION OBJECTIVE

**Fitness Function:** Score = Accuracy - (λ × AvgCalls)

Where:
- Accuracy: Percentage of AIME problems solved correctly (0-100%)
- AvgCalls: Average number of LLM calls per problem
- λ: Efficiency penalty (starts at 0.0, gradually increases to 0.10)

**Goal:** Find routing strategies on the PARETO FRONTIER - maximize accuracy while minimizing LLM calls.

# CONSTRAINTS

- Maximum 10 LLM calls per problem (hard limit, exceeding causes failure)
- Must return 3-digit answer format (0-999) in \\boxed{{}} 
- Minimum 30% accuracy required (below this is heavily penalized)
- Lambda increases over generations, gradually adding efficiency pressure

# ROUTING STRATEGIES TO EXPLORE

1. **Difficulty-based routing:**
   - Use estimate_difficulty() to classify problem
   - Route easy→quick_solve, hard→deep_think

2. **Sequential verification:**
   - Generate solution with one primitive
   - Verify with verify() or self_critique()

3. **Adaptive depth:**
   - Start with quick_solve
   - If uncertain, escalate to deep_think or python_calc

4. **Problem-type detection:**
   - Detect keywords (calculate, prove, find)
   - Route to appropriate primitive

5. **Ensemble strategies:**
   - Use ensemble_vote for medium-difficulty problems
   - Combine multiple primitives for robustness

6. **Computational detection:**
   - If problem involves heavy calculation, use python_calc
   - Otherwise use reasoning-based primitives

7. **Multi-stage pipelines:**
   - First stage: quick_solve or estimate_difficulty
   - Second stage: verify or deep_think based on confidence
   - Final stage: self_critique if calls remaining

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
1. combined_score: The fitness value (accuracy - λ×calls)
2. accuracy: Raw accuracy percentage
3. avg_calls: Average LLM calls per problem
4. text_feedback: Examples of failed problems to learn from
5. lambda_used: Current λ value for this generation

# INNOVATION GUIDELINES

- Be creative! Try unconventional routing strategies
- Use problem features (length, keywords, structure) for routing
- Consider call budget: track remaining calls, adapt strategy
- Combine primitives in novel ways
- Learn from text_feedback about failure patterns
- Balance exploration (new strategies) vs exploitation (refining good strategies)

# CODE REQUIREMENTS

- Only modify the `forward(self, problem)` method inside EVOLVE-BLOCK
- DO NOT change primitive methods (quick_solve, verify, etc.)
- Return (response, total_cost) tuple
- Handle all primitives returning (response, cost) tuples
- Track and sum costs across all primitive calls
- Keep code clean, readable, and well-commented

Remember: The goal is not just high accuracy, but the BEST POINT on the Pareto frontier - optimal balance of accuracy and efficiency!
"""


def create_job_config():
    """Create job configuration for local evaluation."""
    return LocalJobConfig(
        eval_program_path="evaluate.py",
        extra_cmd_args={
            "model_name": "gpt-4o-mini",
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