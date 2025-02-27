import ast
import copy

import astunparse

from .bitvec import BitVec
from .function_parser import FunctionParser


class TweedledumLoopUnroller(ast.NodeTransformer):
    """
    AST transformer for unrolling loops in Tweedledum quantum functions.

    This unroller works with Tweedledum's type system, extracting size
    information from BitVec parameters to unroll loops that iterate over
    their length.
    """

    def __init__(self, symbol_table=None):
        """
        Initialize with a symbol table from Tweedledum's parser.

        Args:
            symbol_table: Dictionary mapping variable names to their types and sizes
                         Format: {var_name: ((type, size), signals)}
        """
        self.symbol_table = symbol_table or {}
        super().__init__()

    def visit_For(self, node):
        """
        Visits For nodes and unrolls them if they iterate over a known range.

        Handles:
        1. Static ranges: for i in range(5)
        2. BitVec length ranges: for i in range(len(bv))
        """
        # Check if we can unroll this loop
        if not self._is_range_call(node.iter):
            return node

        # Extract range parameters
        range_args = self._extract_range_args(node.iter)
        if not range_args:
            return node

        start, stop, step = range_args
        loop_var = node.target.id

        # Process each iteration
        unrolled_body = []
        for i in range(start, stop, step):
            # Create substitution map for this iteration
            substitutions = {loop_var: i}

            # Process each statement in the loop body
            for stmt in node.body:
                # Clone statement to avoid modifying original
                stmt_copy = copy.deepcopy(stmt)

                # Apply variable substitution
                substituter = VariableSubstituter(substitutions, self.symbol_table)
                transformed_stmt = substituter.visit(stmt_copy)

                # Recursively process any nested loops AFTER substitution
                # This ensures substitutions are applied to nested loop bounds
                processed_stmt = self.visit(transformed_stmt)

                # Add statement or unrolled nested loop to result
                if isinstance(processed_stmt, list):
                    unrolled_body.extend(processed_stmt)
                else:
                    unrolled_body.append(processed_stmt)

        return unrolled_body

    def _is_range_call(self, node):
        """Check if a node is a call to range()"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        )

    def _extract_range_args(self, range_call):
        """
        Extract start, stop, and step values from a range() call.

        Handles:
        - range(stop)
        - range(start, stop)
        - range(start, stop, step)
        - range(len(bv)) where bv is a BitVec
        """
        args = range_call.args

        # Try to evaluate each argument to a constant
        evaluated_args = []

        for arg in args:
            # If argument is a len() call, handle specially
            if self._is_len_call(arg):
                bitvec_size = self._get_bitvec_size(arg.args[0])
                if bitvec_size is not None:
                    evaluated_args.append(bitvec_size)
                    continue

            # Otherwise try to evaluate the expression
            substituter = VariableSubstituter({}, self.symbol_table)
            value = substituter._evaluate_expr(arg)
            if value is not None:
                evaluated_args.append(value)
            else:
                return None  # Can't evaluate all arguments

        # Handle different number of arguments
        if len(evaluated_args) == 1:
            return 0, evaluated_args[0], 1
        elif len(evaluated_args) == 2:
            return evaluated_args[0], evaluated_args[1], 1
        elif len(evaluated_args) == 3:
            return evaluated_args[0], evaluated_args[1], evaluated_args[2]

        return None

    def _is_len_call(self, node):
        """Check if a node is a call to len()"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
        )

    def _get_bitvec_size(self, node):
        """
        Get the size of a BitVec variable from the symbol table.

        Args:
            node: The AST node representing the BitVec variable

        Returns:
            int: The size of the BitVec, or None if not found
        """
        if isinstance(node, ast.Name) and node.id in self.symbol_table:
            # In Tweedledum's symbol table format: {var: ((type, size), signals)}
            var_info = self.symbol_table[node.id]
            if var_info and var_info[0] and len(var_info[0]) >= 2:
                return var_info[0][1]  # Extract size from (type, size) tuple
        return None


class VariableSubstituter(ast.NodeTransformer):
    def __init__(self, substitutions, symbol_table=None):
        self.substitutions = substitutions
        self.symbol_table = symbol_table or {}
        super().__init__()

    def _evaluate_expr(self, expr_node):
        """Evaluate an expression node to a constant if possible"""
        # Handle constants directly
        if isinstance(expr_node, ast.Constant):
            return expr_node.value

        # Handle names that are in our substitution table
        if isinstance(expr_node, ast.Name) and expr_node.id in self.substitutions:
            return self.substitutions[expr_node.id]

        # Handle binary operations
        if isinstance(expr_node, ast.BinOp):
            left_val = self._evaluate_expr(expr_node.left)
            right_val = self._evaluate_expr(expr_node.right)

            if left_val is not None and right_val is not None:
                try:
                    return self._evaluate_binop(left_val, expr_node.op, right_val)
                except:
                    pass

        # Handle len() calls
        if self._is_len_call(expr_node) and len(expr_node.args) == 1:
            arg_node = expr_node.args[0]
            if isinstance(arg_node, ast.Name) and arg_node.id in self.symbol_table:
                var_info = self.symbol_table[arg_node.id]
                if var_info and var_info[0] and len(var_info[0]) >= 2:
                    return var_info[0][1]  # Return size

        return None  # Can't evaluate to a constant

    def _is_len_call(self, node):
        """Check if a node is a call to len()"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
        )

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.substitutions:
            # Replace with constant
            const_node = ast.Constant(value=self.substitutions[node.id])
            ast.copy_location(const_node, node)
            return const_node
        return node

    def visit_Call(self, node):
        # Keep len() calls but process their arguments
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            # Don't replace len() itself, but still process its arguments
            node.args = [self.visit(arg) for arg in node.args]
            return node

        # For other calls, process arguments
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_BinOp(self, node):
        # Track if this expression contains any loop variables
        has_loop_vars = self._contains_loop_vars(node)

        # Visit left and right sides
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Try direct evaluation
        result_val = self._evaluate_expr(node)
        if result_val is not None and has_loop_vars:
            # If we got a value and it contains loop variables, replace with constant
            const_node = ast.Constant(value=result_val)
            ast.copy_location(const_node, node)
            return const_node

        # If not, return the modified node
        node.left = left
        node.right = right
        return node

    def _contains_loop_vars(self, node):
        """Check if an expression contains any loop variables"""
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and subnode.id in self.substitutions:
                return True
        return False

    def _evaluate_binop(self, left, op, right):
        """Evaluate a binary operation on constants"""
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        # Add other operations as needed
        raise ValueError(f"Unsupported operation: {type(op)}")


class UnrollingFunctionParser(FunctionParser):
    """
    Extended FunctionParser that unrolls loops in quantum functions.
    """

    def __init__(self, source):
        # Initialize
        self._symbol_table = {}
        self._parameters_signature = []
        self._return_signature = []

        # Parse source
        original_ast = ast.parse(source)

        # Extract type info
        self._extract_type_info(original_ast)

        # Unroll all loops
        unroller = TweedledumLoopUnroller(self._symbol_table)
        unrolled_ast = unroller.visit(copy.deepcopy(original_ast))

        # Fix AST
        ast.fix_missing_locations(unrolled_ast)

        # Get unrolled source
        unrolled_source = astunparse.unparse(unrolled_ast)
        print("DEBUG: Unrolled source:")
        print(unrolled_source)

        # Make sure we've completely unrolled all loops
        # Check if there are any remaining For nodes
        has_loops = False
        for node in ast.walk(unrolled_ast):
            if isinstance(node, ast.For):
                has_loops = True
                print(f"WARNING: Loop not fully unrolled: {astunparse.unparse(node)}")

        if has_loops:
            print("Some loops could not be unrolled!")
            # Add loop variables to symbol table
            self._add_loop_variables(unrolled_ast)

        # Initialize parent with fully unrolled code
        super().__init__(unrolled_source)

    def _extract_type_info(self, tree):
        """Extract type information from function signature"""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Store function name
                self.name = node.name

                # Process arguments
                for arg in node.args.args:
                    if arg.annotation and isinstance(arg.annotation, ast.Call):
                        arg_type = arg.annotation.func.id
                        if arg_type == "BitVec" and len(arg.annotation.args) >= 1:
                            size = ast.literal_eval(arg.annotation.args[0])
                            self._symbol_table[arg.arg] = (
                                (type(BitVec(1)), size),
                                None,
                            )

    def _add_loop_variables(self, tree):
        """Add loop variables to symbol table"""
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                loop_var = node.target.id
                if loop_var not in self._symbol_table:
                    # Add loop variable as BitVec(1)
                    self._symbol_table[loop_var] = ((type(BitVec(1)), 1), None)
