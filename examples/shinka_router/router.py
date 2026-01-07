"""
Router representation with executable routing logic.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Router:
    """Routing policy that decides which model to use for each problem."""
    
    def __init__(self, logic: str, name: str = "unnamed_router"):
        """Initialize router with executable Python code.
        
        Args:
            logic: Python code string that implements routing logic
            name: Human-readable name for this router
        """
        self.logic = logic
        self.name = name
        self._compiled = None
        
    def route(self, problem: str, context: Dict[str, Any]) -> str:
        """Execute routing logic to select a model.
        
        Args:
            problem: The problem text to route
            context: Additional context (features, metadata, etc.)
            
        Returns:
            Model name string ('gpt-4o-mini', 'gpt-4.1-nano', or 'o4-mini')
        """
        # Compile logic once for efficiency
        if self._compiled is None:
            try:
                self._compiled = compile(self.logic, '<router>', 'exec')
            except SyntaxError as e:
                logger.error(f"Router compilation failed: {e}")
                return 'gpt-4o-mini'  # Default fallback
        
        # Execute routing logic with access to problem and context
        namespace = {
            'problem': problem,
            'context': context,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'min': min,
            'max': max,
            'sum': sum,
        }
        
        try:
            exec(self._compiled, namespace)
            model = namespace.get('model', 'gpt-4o-mini')
            
            # Validate model name
            valid_models = ['gpt-4o-mini', 'gpt-4.1-nano', 'o4-mini']
            if model not in valid_models:
                logger.warning(f"Invalid model '{model}', using gpt-4o-mini")
                return 'gpt-4o-mini'
            
            return model
        except Exception as e:
            logger.error(f"Router execution failed: {e}")
            return 'gpt-4o-mini'  # Safe fallback
    
    def to_dict(self) -> Dict[str, str]:
        """Serialize router for storage."""
        return {
            'logic': self.logic,
            'name': self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Router':
        """Deserialize router from storage."""
        return cls(logic=data['logic'], name=data.get('name', 'unnamed'))
    
    def __repr__(self) -> str:
        return f"Router(name='{self.name}')"