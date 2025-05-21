import ast
import logging
from typing import Any, Dict, Optional

# logger named 'classical_expression_evaluator'
logger = logging.getLogger(__name__)


class ClassicalExpressionEvaluator:
    """
    Evaluates classical expressions using Python's built-in capabilities.
    """

    def __init__(self, variables: Dict[str, Any]):
        self.variables = variables

    def evaluate(self, node) -> Optional[Any]:
        """
        Evaluate an AST node to a constant value if possible.
        Returns None if the expression cannot be statically evaluated.
        """
        try:
            # Convert the AST node to source code
            source = ast.unparse(node)
            logger.debug(f"ClassicalEvaluator: {source}")

            # Create a safe environment with only our variables
            # This prevents arbitrary code execution
            safe_globals = {"__builtins__": {}}

            # Add only needed built-ins
            for name in ["len", "sum", "min", "max", "abs", "all", "any"]:
                safe_globals[name] = __builtins__[name]

            # Evaluate the expression in our controlled environment
            return eval(source, safe_globals, self.variables)
        except Exception as e:
            # For index errors and similar, propagate the exception
            # so it can be caught and handled by the transformer
            if isinstance(e, (IndexError, KeyError, TypeError)):
                raise
            # For other errors, just return None to indicate
            # we can't evaluate statically
            return None
