import ast

from typing import Any, List

import astunparse

from .function_parser import ParseError

from .meta_fns import _global_generators


class TweedledumMetaInliner(ast.NodeTransformer):
    """Simple AST transformer for inlining generator function calls."""

    def __init__(self, generator_funcs=_global_generators):
        self.generator_funcs = generator_funcs or {}
        super().__init__()

    def visit_Expr(self, node):
        """Handle expression statements that contain generator calls."""
        # Only process expression statements with generator function calls
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self.generator_funcs
        ):
            try:
                # Call generator function with the AST node arguments
                generator_func = self.generator_funcs[node.value.func.id]
                statements = generator_func(*node.value.args)

                # Simply return the list of statements to replace this node
                return statements
            except Exception as e:
                raise ValueError(
                    f"Error in generator '{node.value.func.id}': {str(e)}"
                ) from e

        # Not a generator call, continue normal processing
        return self.generic_visit(node)
