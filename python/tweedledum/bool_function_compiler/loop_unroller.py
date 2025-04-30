# bool_function_compiler/loop_unroller.py
import ast
import copy
from typing import Any, Dict, List, Union

# We need the evaluator to resolve range arguments and potentially iterables
from .classical_expression_evaluator import ClassicalExpressionEvaluator


# --- LoopVariableReplacer (Helper for the Unroller) ---
# This version substitutes variables with Constants holding their values
class LoopVariableReplacer(ast.NodeTransformer):
    """
    Replaces references to loop variables with their constant values during unrolling.
    Handles non-integer constants.
    """

    def __init__(self, substitutions: Dict[str, Any]):
        self.substitutions = substitutions
        super().__init__()

    def visit_Name(self, node):
        """Replace loop variable references with constant values."""
        if node.id in self.substitutions and isinstance(node.ctx, ast.Load):
            new_node = ast.Constant(value=self.substitutions[node.id])
            ast.copy_location(new_node, node)
            return new_node
        return node


# --- The Main Classical Loop Unroller Pass ---
class ClassicalLoopUnroller(ast.NodeTransformer):
    """
    AST transformer that unrolls 'for' loops iterating over classical ranges
    or known classical lists/tuples.

    Requires classical input values to resolve iterables.
    RESTRICTION: Loop target must be a single variable name (no tuple unpacking).
    """

    def __init__(self, classical_inputs: Dict[str, Any]):
        self.classical_inputs = classical_inputs
        # Create an evaluator instance internally or receive one if needed elsewhere
        self.evaluator = ClassicalExpressionEvaluator(self.classical_inputs)
        super().__init__()

    def visit_For(self, node: ast.For) -> Union[List[ast.AST], ast.For]:
        """
        Visits For nodes and unrolls them if they iterate over a classical
        range or a known classical list/tuple.
        """
        loop_iterable_val = None
        iterable_node = node.iter
        loop_target = node.target

        # Ensure loop target is a simple name before proceeding
        if not isinstance(loop_target, ast.Name):
            target_repr = ast.dump(loop_target)
            # If we don't unroll, just visit the body and return the node
            # Or raise error if unrolling is mandatory for compatible code downstream
            # For now, let's raise, assuming downstream requires unrolling.
            raise ValueError(
                f"Loop target must be a simple variable name for unrolling, got {target_repr}"
            )

        # 1. Determine the iterable value if classical
        if (
            isinstance(iterable_node, ast.Call)
            and isinstance(iterable_node.func, ast.Name)
            and iterable_node.func.id == "range"
        ):
            # --- Handle classical range() ---
            try:
                range_args = []
                for arg in iterable_node.args:
                    val = self.evaluator.evaluate(arg)  # Use internal evaluator
                    if val is None:
                        raise ValueError(
                            f"Cannot evaluate range argument: {ast.dump(arg)}"
                        )
                    if not isinstance(val, int):
                        raise ValueError(
                            f"Range arguments must evaluate to integers, got: {type(val)}"
                        )
                    range_args.append(val)

                if len(range_args) == 1:
                    loop_iterable_val = range(range_args[0])
                elif len(range_args) == 2:
                    loop_iterable_val = range(range_args[0], range_args[1])
                elif len(range_args) == 3:
                    loop_iterable_val = range(
                        range_args[0], range_args[1], range_args[2]
                    )
                else:
                    raise ValueError("Invalid number of arguments to range()")
            except Exception as e:
                # If range cannot be evaluated, maybe keep the loop? Or raise?
                # Raising for now, assumes loops must be unrolled.
                raise ValueError(
                    f"Failed to evaluate classical range for unrolling: {e}"
                ) from e

        elif (
            isinstance(iterable_node, ast.Name)
            and iterable_node.id in self.classical_inputs
        ):
            # --- Handle classical variable (list/tuple) ---
            potential_iterable = self.classical_inputs[iterable_node.id]
            if isinstance(potential_iterable, (list, tuple)):
                loop_iterable_val = potential_iterable
            # else: variable is classical but not iterable - cannot unroll

        elif isinstance(iterable_node, ast.Constant) and isinstance(
            iterable_node.value, (list, tuple)
        ):
            # --- Handle constant list/tuple (e.g., from previous substitution) ---
            loop_iterable_val = iterable_node.value

        # 2. If we determined a classical iterable, unroll the loop
        if loop_iterable_val is not None:
            unrolled_body = []
            for current_loop_val in loop_iterable_val:
                # Create substitution dictionary for the single loop variable
                substitutions = {loop_target.id: current_loop_val}
                replacer = LoopVariableReplacer(substitutions)

                for stmt in node.body:
                    stmt_copy = copy.deepcopy(stmt)
                    # Apply substitution for this iteration's values
                    transformed_stmt = replacer.visit(stmt_copy)
                    # Recursively visit the transformed statement *using this unroller*
                    # This handles nested loops correctly.
                    processed_stmt_or_list = self.visit(transformed_stmt)

                    # Add the processed statement(s) to the unrolled body
                    if isinstance(processed_stmt_or_list, list):
                        unrolled_body.extend(processed_stmt_or_list)
                    elif processed_stmt_or_list is not None:
                        unrolled_body.append(processed_stmt_or_list)
            # Return the list of unrolled statements, replacing the original For node
            return unrolled_body
        else:
            # If iterable wasn't a known classical list/tuple or range,
            # either raise an error or return the node unchanged.
            # Raising error assumes loops must be unrolled for downstream compatibility.
            try:
                iter_repr = ast.unparse(iterable_node)
            except:
                iter_repr = ast.dump(iterable_node)
            raise ValueError(
                f"Cannot unroll loop: iterable '{iter_repr}' must be a classical range() or a known classical list/tuple."
            )

        # If we choose not to raise an error above for non-unrollable loops,
        # we would process the body normally and return the original node:
        # node.body = [self.visit(stmt) for stmt in node.body]
        # return node
