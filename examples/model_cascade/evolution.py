"""
Evolution engine for router logic with mutation operators.
"""
import logging
import random
from typing import List, Optional
from .router import Router

logger = logging.getLogger(__name__)


class RouterEvolution:
    """Evolves routing logic using mutation operators."""
    
    INITIAL_ROUTERS = [
        # Length-based thresholds
        """
# Simple length threshold
if len(problem) > 200:
    model = 'o4-mini'
elif len(problem) > 100:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""",
        # Keyword-based routing
        """
# Keyword complexity routing
keywords = ['prove', 'theorem', 'integral', 'derivative', 'limit']
complexity = sum(1 for kw in keywords if kw in problem.lower())
if complexity >= 2:
    model = 'o4-mini'
elif complexity >= 1:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""",
        # Question mark heuristic
        """
# Multi-step detection
question_marks = problem.count('?')
if question_marks >= 2:
    model = 'o4-mini'
elif question_marks == 1 and len(problem) > 150:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""",
        # Number density
        """
# Number density routing
import re
numbers = len(re.findall(r'\\d+', problem))
density = numbers / max(len(problem.split()), 1)
if density > 0.2:
    model = 'o4-mini'
elif density > 0.1:
    model = 'gpt-4.1-nano'
else:
    model = 'gpt-4o-mini'
""",
    ]
    
    def __init__(self, llm_client, system_prompt: str, mutation_prompt: str):
        """Initialize evolution engine.
        
        Args:
            llm_client: LLM client for generating mutations
            system_prompt: System prompt for LLM
            mutation_prompt: Template for mutation requests
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.mutation_prompt = mutation_prompt
        
    def initialize_population(self, size: int = 4) -> List[Router]:
        """Create initial router population.
        
        Args:
            size: Population size
            
        Returns:
            List of Router instances
        """
        routers = []
        for i, logic in enumerate(self.INITIAL_ROUTERS[:size]):
            router = Router(logic=logic.strip(), name=f"init_{i}")
            routers.append(router)
        return routers
    
    def mutate_router(
        self,
        router: Router,
        inspiration_routers: Optional[List[Router]] = None,
    ) -> Optional[Router]:
        """Generate a mutated version of a router.
        
        Args:
            router: Parent router to mutate
            inspiration_routers: Additional routers for context
            
        Returns:
            New Router instance or None if generation fails
        """
        # Build prompt with parent logic
        prompt = self.mutation_prompt.format(
            parent_logic=router.logic,
            parent_name=router.name,
        )
        
        # Add inspiration routers if provided
        if inspiration_routers:
            inspiration_text = "\n\n# Inspiration Routers:\n"
            for i, insp in enumerate(inspiration_routers[:3]):
                inspiration_text += f"\n## Router {i+1} ({insp.name}):\n```python\n{insp.logic}\n```\n"
            prompt += inspiration_text
        
        # Query LLM for mutation
        try:
            result = self.llm_client.query(
                msg=prompt,
                system_msg=self.system_prompt,
            )
            
            if result is None:
                logger.error("LLM query returned None")
                return None
            
            # Extract code between ```python and ```
            content = result.content
            if '```python' in content:
                code_start = content.find('```python') + 9
                code_end = content.find('```', code_start)
                logic = content[code_start:code_end].strip()
            elif '```' in content:
                code_start = content.find('```') + 3
                code_end = content.find('```', code_start)
                logic = content[code_start:code_end].strip()
            else:
                logic = content.strip()
            
            # Extract name if present
            name = "mutated"
            if '<NAME>' in content and '</NAME>' in content:
                name_start = content.find('<NAME>') + 6
                name_end = content.find('</NAME>')
                name = content[name_start:name_end].strip()
            
            return Router(logic=logic, name=name)
            
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            return None
    
    def crossover(
        self,
        router1: Router,
        router2: Router,
    ) -> Optional[Router]:
        """Combine two routers via LLM-based crossover.
        
        Args:
            router1: First parent router
            router2: Second parent router
            
        Returns:
            New Router combining aspects of both parents
        """
        prompt = f"""
Create a new routing algorithm that combines the best aspects of these two routers:

# Router 1 ({router1.name}):
```python
{router1.logic}
```

# Router 2 ({router2.name}):
```python
{router2.logic}
```

Create a novel combination that uses ideas from both. Output only the Python code.
"""
        
        try:
            result = self.llm_client.query(
                msg=prompt,
                system_msg=self.system_prompt,
            )
            
            if result is None:
                return None
            
            # Extract code
            content = result.content
            if '```python' in content:
                code_start = content.find('```python') + 9
                code_end = content.find('```', code_start)
                logic = content[code_start:code_end].strip()
            else:
                logic = content.strip()
            
            name = f"cross_{router1.name}_{router2.name}"
            return Router(logic=logic, name=name)
            
        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            return None