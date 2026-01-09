# Best Router: mutated

# <n>AIME_Router_Improved</n>
import re

# Define mathematical keywords, complexity indicators, and multi-step indicators
keywords = ['prove', 'show', 'find', 'determine', 'compute', 'integral', 'derivative', 'theorem', 'limit', 'function', 'solve', 'calculate']
multi_step_indicators = ['and', 'or', 'then', 'if', 'since', 'more than', 'less than']
equation_density_threshold = 0.15
complex_domains = ['geometry', 'combinatorics', 'number theory', 'algebra', 'probability']

# Calculate keyword complexity
kw_count = sum(1 for kw in keywords if kw in problem.lower())

# Check for complex domains
domain_count = sum(1 for domain in complex_domains if domain in problem.lower())

# Calculate number density
numbers = len(re.findall(r'\d+', problem))
number_density = numbers / max(len(problem.split()), 1)

# Calculate equation density
equation_count = len(re.findall(r'=', problem)) + len(re.findall(r'\+', problem)) + len(re.findall(r'-', problem)) + len(re.findall(r'\*', problem)) + len(re.findall(r'/', problem))
equation_density = equation_count / max(len(problem.split()), 1)

# Count multi-step indicators
multi_step_count = sum(1 for indicator in multi_step_indicators if indicator in problem.lower())

# Routing logic enhanced for better identification of complex problems
if kw_count >= 4 or domain_count > 1 or (kw_count >= 3 and (equation_density > equation_density_threshold or multi_step_count >= 2)):
    model = 'o4-mini'
elif kw_count >= 2 or number_density > 0.1 or (equation_density > 0.1 and multi_step_count >= 1):
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'