# bool_function_compiler/transformer.py

import ast
from typing import Any, Dict, List, Optional, Set, Union

# Assuming these are correctly importable from your package structure
from .classical_expression_evaluator import ClassicalExpressionEvaluator
from .variable_classifier import VariableClassifier

import logging

logger = logging.getLogger(__name__)


class QuantumCircuitTransformer(ast.NodeTransformer):
    """
    Transforms the AST after meta-function inlining and loop unrolling.
    Focuses on classical value substitution, removing purely classical logic
    (assignments, conditional branches), and simplifying classical expressions.
    Assumes loops have already been unrolled by a preceding pass.
    """

    def __init__(
        self,
        classical_inputs: Dict[str, Any],
        quantum_params: Dict[str, Any],
        # globals_dict might be needed if evaluator needs access to more functions
        globals_dict: Dict,
        used_names: Set[str],
    ):
        # Use a mutable copy of classical inputs for state updates
        self.classical_inputs = dict(classical_inputs)
        self.quantum_params = quantum_params
        self.globals = globals_dict or {}
        self.used_names = used_names or set()

        for name in quantum_params:
            self.used_names.add(name)

        # Classifier can run first for initial analysis
        self.classifier = VariableClassifier(quantum_params, self.classical_inputs)

        # Pass the mutable dictionary to the evaluator
        self.evaluator = ClassicalExpressionEvaluator(self.classical_inputs)

        super().__init__()

    def transform(self, tree: ast.AST) -> ast.AST:
        """
        Apply the classical/quantum separation transformation to an AST
        (assuming loops are already unrolled).
        """
        # Run classifier on the input tree (which should be unrolled)
        self.classifier.visit(tree)
        self.classifier.summarize()

        # Transform the AST; state updates happen during visit
        new_tree = self.visit(tree)

        # Ensure the final tree is valid
        ast.fix_missing_locations(new_tree)
        return new_tree

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Transforms the function definition:
        - Updates signature to only include quantum parameters.
        - Processes the body, handling removed nodes (returning None/[]).
        - Removes decorators.
        """
        # Create new argument list keeping only quantum parameters
        new_args = []
        processed_quantum_params = {}  # Store actual types/values for reference if needed
        for name, spec in self.quantum_params.items():
            # Note: Lambda evaluation should happen *before* the transformer runs.
            # Here, we just construct the AST for the signature.
            # Assuming 'spec' here is the evaluated BitVec type/object.
            # TODO: Ensure 'spec' passed to __init__ is the processed value, not a lambda.
            # For now, just use the name. Annotation might need adjustment.
            # A better approach might be to get the final quantum types from
            # the processed_quantum_params generated before calling the transformer.
            arg_annotation = ast.Constant(
                value=spec
            )  # Placeholder, might need better annotation later
            arg = ast.arg(arg=name, annotation=arg_annotation)
            new_args.append(arg)

        node.args.args = new_args
        node.args.defaults = []  # Clear defaults
        node.args.kw_defaults = []  # Clear kw defaults
        node.args.kwarg = None
        node.args.posonlyargs = []
        node.args.vararg = None
        # Clear type comment if any
        node.type_comment = None

        # Process function body statements, correctly handling node removals
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            # visit can return: a single node, None (remove), or a list (from If)
            if isinstance(result, list):
                new_body.extend(
                    result
                )  # Add all statements if visit_If returned a list
            elif result is not None:
                new_body.append(result)  # Add single node, skip if None
        node.body = new_body

        # Remove decorators and return type annotation (as it might be misleading after transform)
        node.decorator_list = []

        return node

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.Assign]:
        """
        Process assignment statements.
        Evaluate RHS; if purely classical, update state and REMOVE the node.
        If quantum, keep the node.
        """
        # Process right side first - this might simplify it
        node.value = self.visit(node.value)

        # Try to evaluate the (potentially simplified) right-hand side
        evaluated_value = self.evaluator.evaluate(node.value)

        # Check if assignment target is a simple name
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            target_name = target.id

            if evaluated_value is not None:
                # --- RHS is purely classical ---
                # Update the dictionary used by the evaluator so subsequent
                # uses of this variable *within this transformer pass* resolve correctly.
                logger.info(
                    f"Transformer Update: Assigning classical value {evaluated_value} to '{target_name}' (removing Assign node)"
                )
                self.classical_inputs[target_name] = evaluated_value

                # Remove this classical assignment node from the final AST
                return None  # Returning None removes the node
            else:
                # --- RHS involves quantum variables ---
                # Keep the assignment node as it's part of the quantum logic.
                self.used_names.add(target_name)
                # If target was previously classical, remove stale value
                if target_name in self.classical_inputs:
                    logger.info(
                        f"Transformer Info: Variable '{target_name}' reassigned quantum value, removing from classical state."
                    )
                    del self.classical_inputs[target_name]
                # Return the processed node (keep it in the AST)
                return node
        else:
            # Handle more complex targets (e.g., quantum subscript assign: bitvec[idx] = ...)
            # These likely involve quantum ops or targets, so keep them.
            node.targets = [self.visit(target) for target in node.targets]
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.used_names.add(target.id)
            # Keep the node
            return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Handle function calls."""
        # Process arguments first
        node.args = [self.visit(arg) for arg in node.args]
        # Assuming kwargs are not used or handled if necessary
        node.keywords = [self.visit(kw) for kw in node.keywords]

        if isinstance(node.func, ast.Name):
            # Handle BitVec constructor specifically if needed for analysis/simplification
            if node.func.id == "BitVec":
                return self._process_bitvec_constructor(node)
            # Add handling for other known classical/quantum functions if required
            # E.g., len() on a classical list could be evaluated:
            # if node.func.id == 'len' and len(node.args) == 1:
            #     arg_val = self.evaluator.evaluate(node.args[0])
            #     if isinstance(arg_val, (list, tuple, str)):
            #          return ast.Constant(value=len(arg_val))

        # Default: assume it's a quantum function call or unknown, keep node
        return node

    def _process_bitvec_constructor(self, node: ast.Call) -> ast.Call:
        """Handle BitVec constructor calls, evaluate size if possible."""
        # First argument should be size
        if len(node.args) >= 1:
            size_arg = node.args[0]
            # Ensure we visit the arg node first, it might become constant
            visited_size_arg = self.visit(size_arg)
            node.args[0] = visited_size_arg  # Update arg list
            if not isinstance(visited_size_arg, ast.Constant):
                # If still not constant, try evaluating it
                size_value = self.evaluator.evaluate(visited_size_arg)
                if size_value is not None and isinstance(size_value, int):
                    node.args[0] = ast.Constant(value=size_value)
            # Can also evaluate the optional second argument if needed
            if len(node.args) >= 2:
                node.args[1] = self.visit(node.args[1])

        return node

    def visit_If(self, node: ast.If) -> Optional[Union[ast.If, List[ast.AST]]]:
        """
        Pre-evaluate conditionals with classical values. Remove or replace
        with the appropriate branch body.
        """
        # Visit the test expression first - it might get simplified
        test_node = self.visit(node.test)

        # Try to evaluate the (potentially simplified) condition
        logger.info(
            f"Transformer: Evaluating IF condition: {ast.dump(test_node)}"
        )  # Use dump for clarity
        condition_value = self.evaluator.evaluate(test_node)
        logger.info(f"Transformer: Condition evaluated to: {condition_value}")

        if condition_value is not None:
            # --- Static evaluation succeeded ---
            if condition_value:  # Condition is True, process and return body
                logger.info("Transformer: IF condition True, processing body.")
                new_body = []
                for stmt in node.body:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_body.extend(result)
                    elif result is not None:
                        new_body.append(result)
                return new_body  # Return list of statements
            elif node.orelse:  # Condition is False, process and return orelse
                logger.info("Transformer: IF condition False, processing orelse.")
                new_orelse = []
                for stmt in node.orelse:
                    result = self.visit(stmt)
                    if isinstance(result, list):
                        new_orelse.extend(result)
                    elif result is not None:
                        new_orelse.append(result)
                return new_orelse  # Return list of statements
            else:  # Condition is False, no orelse, remove the If node
                logger.info(
                    "Transformer: IF condition False, no orelse, removing node."
                )
                return None  # Return None to remove
        else:
            # --- Cannot evaluate classically ---
            # This implies the condition involves quantum variables.
            # Raise error as quantum conditions are not supported in standard If.
            condition_str = ast.dump(test_node)
            raise ValueError(
                f"Condition '{condition_str}' could not be evaluated classically and quantum conditions are not supported in 'if' statements."
            )

    def visit_BinOp(self, node: ast.BinOp) -> Union[ast.BinOp, ast.Constant]:
        """Process binary operations. Evaluate if purely classical."""
        # Visit operands first
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        # Try to evaluate the whole operation
        result = self.evaluator.evaluate(node)
        if result is not None:
            # This is a purely classical operation - replace with constant
            return ast.Constant(value=result)
        else:
            # Involves quantum variables, keep the operation node
            return node

    def visit_Compare(self, node: ast.Compare) -> Union[ast.Compare, ast.Constant]:
        """Process comparison operations. Evaluate if purely classical."""
        # Visit operands first
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]

        # Try to evaluate the whole comparison
        result = self.evaluator.evaluate(node)
        if result is not None:
            # Purely classical comparison
            return ast.Constant(value=result)
        else:
            # Involves quantum variables, keep the comparison node
            return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[ast.UnaryOp, ast.Constant]:
        """Process unary operations. Evaluate if purely classical."""
        # Visit operand first
        node.operand = self.visit(node.operand)

        # Try to evaluate the whole operation
        result = self.evaluator.evaluate(node)
        if result is not None:
            # Purely classical operation (e.g., not True)
            return ast.Constant(value=result)
        else:
            # Quantum operation (e.g., ~BitVec), keep the node
            return node

    def visit_Subscript(
        self, node: ast.Subscript
    ) -> Union[ast.Subscript, ast.Constant]:
        """Handle subscript operations. Evaluate if purely classical."""
        # Visit the index/slice first
        node.slice = self.visit(node.slice)
        # Visit the value being indexed
        node.value = self.visit(node.value)

        # Try to evaluate the whole subscript expression classically
        evaluated_value = self.evaluator.evaluate(node)
        if evaluated_value is not None:
            # Classical subscript succeeded (e.g., list[index], tuple[index])
            return ast.Constant(value=evaluated_value)
        else:
            # If it couldn't be fully evaluated (e.g., quantum_vec[classical_index]),
            # return the potentially simplified subscript node
            return node

    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Constant]:
        """Process variable references. Substitute if classical."""
        if isinstance(node.ctx, ast.Load):
            # Check if this is a reference to a known classical variable
            if node.id in self.classical_inputs:
                # Replace the name node with a constant node holding its value
                return ast.Constant(value=self.classical_inputs[node.id])
        # If not loading a known classical variable, or if it's a Store context,
        # return the node as is
        return node
