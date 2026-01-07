# Best Router: mutated

# Improved Router for AIME Problems <improved_router_v2>
import re

# Define mathematical keywords with weights
keywords = {
    'prove': 3,
    'show': 3,
    'find': 2,
    'determine': 2,
    'compute': 1,
    'integral': 2,
    'derivative': 2,
    'theorem': 3,
    'limit': 2,
    'factor': 1,
    'solve': 2,
    'graph': 1,
    'conjecture': 3,
    'evaluate': 2,
}

# Count keyword occurrences and their weights
kw_count = sum(keywords[kw] for kw in keywords if kw in problem.lower())

# Count the number of equations and their density
equation_count = len(re.findall(r'=', problem))
number_count = len(re.findall(r'\d+', problem))
equation_density = equation_count / max(len(problem.split()), 1)

# Count the number of multi-step indicators (e.g., "first," "next," "finally")
multi_step_indicators = ['first', 'next', 'finally', 'then', 'after']
multi_step_count = sum(1 for indicator in multi_step_indicators if indicator in problem.lower())

# Determine model based on keyword significance, equation density, and multi-step indicators
if kw_count >= 7 or equation_density > 0.3 or multi_step_count >= 2:
    model = 'o4-mini'
elif kw_count >= 4 or equation_density > 0.2 or multi_step_count >= 1:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'