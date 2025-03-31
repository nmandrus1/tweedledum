import ast
import copy
import inspect
from typing import Any, Dict, Optional, Set

from .classical_expression_evaluator import ClassicalExpressionEvaluator
from .variable_classifier import VariableClassifier


class QuantumCircuitTransformer(ast.NodeTransformer):
    """
    Enhanced transformer with function calling capability.
    """

    def __init__(
        self,
        classical_inputs: Dict[str, Any],
        quantum_params: Dict[str, Any],
        globals_dict: Dict,
        used_names: Set[str],
    ):
        self.classical_inputs = classical_inputs
        self.quantum_params = quantum_params
        self.globals = globals_dict or {}
        self.used_names = used_names or set()

        # Add quantum params to used names
        for name in quantum_params:
            self.used_names.add(name)

        # Classify variables
        self.classifier = VariableClassifier(quantum_params, classical_inputs)

        # Create expression evaluator
        self.evaluator = ClassicalExpressionEvaluator(classical_inputs)

        # Create function call handler
        self.func_handler = FunctionCallHandler(self.globals)

        # Track inlined function calls and their results
        self.inlined_functions = []

        super().__init__()

    def transform(self, tree: ast.AST) -> ast.AST:
        """
        Apply the full transformation pipeline to an AST.
        """
        # First classify all variables
        self.classifier.visit(tree)
        self.classifier.summarize()

        # Then transform the AST
        new_tree = self.visit(tree)

        # Fix line numbers and context references
        ast.fix_missing_locations(new_tree)

        return new_tree

    def visit_FunctionDef(self, node):
        """Transform the function definition to include quantum parameters."""
        # Create new argument list that keeps quantum parameters
        new_args = []

        for arg in node.args.args:
            print(arg)

        print(self.quantum_params)
        # Check if parameters match quantum_params
        for name, value in self.quantum_params.items():
            # Keep quantum parameters in the signature
            arg = ast.arg(arg=name, annotation=ast.Constant(value=value))
            new_args.append(arg)

        # Update function signature
        node.args.args = new_args

        # Process function body
        node.body = [self.visit(stmt) for stmt in node.body]

        # remove decorators
        node.decorator_list = []

        return node

    def visit_Assign(self, node):
        """Process assignment statements."""
        # Process right side
        node.value = self.visit(node.value)

        # Process targets
        node.targets = [self.visit(target) for target in node.targets]

        # Update used names
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.used_names.add(target.id)

        return node

    def visit_Call(self, node):
        """Handle function calls, including quantum function calls."""
        # Process arguments first
        node.args = [self.visit(arg) for arg in node.args]

        if isinstance(node.func, ast.Name):
            # Handle BitVec constructor specially
            if node.func.id == "BitVec":
                return self._process_bitvec_constructor(node)

            # Check if this is a quantum function call
            if node.func.id in self.globals and callable(self.globals[node.func.id]):
                func = self.globals[node.func.id]
                if hasattr(func, "_quantum_params"):
                    # Process the function call
                    result = self.func_handler.process_call(
                        node, self.evaluator, self.used_names
                    )
                    if result:
                        # Add to inlined functions list
                        func_name = node.func.id
                        result_var = result.id
                        # Get transformed function
                        from transform_function import transform_function

                        _, func_obj = transform_function(func, {}, self.quantum_params)
                        # Add to inlined functions
                        self.inlined_functions.append((func_name, result_var, func_obj))
                        return result

        return node

    def _process_bitvec_constructor(self, node):
        """Handle BitVec constructor calls."""
        # Process BitVec constructor arguments
        # First argument should be size
        if len(node.args) >= 1:
            size_arg = node.args[0]
            if not isinstance(size_arg, ast.Constant):
                # Try to evaluate size
                size_value = self.evaluator.evaluate(size_arg)
                if size_value is not None:
                    node.args[0] = ast.Constant(value=size_value)

        return node

    def visit_If(self, node):
        """Pre-evaluate conditionals with classical values."""
        # Try to evaluate the condition
        condition_value = self.evaluator.evaluate(node.test)

        if condition_value is not None:
            # Static evaluation succeeded
            if condition_value:
                # Return transformed body
                return [self.visit(stmt) for stmt in node.body]
            elif node.orelse:
                # Return transformed else body
                return [self.visit(stmt) for stmt in node.orelse]
            else:
                # Skip this branch entirely
                return []

        # Cannot evaluate - this is likely a quantum condition
        raise ValueError("Quantum conditions are not supported")

    def visit_For(self, node):
        """Pre-evaluate for loops with fixed ranges."""
        # Check if the loop can be unrolled
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            # Try to evaluate range arguments
            range_args = []

            for arg in node.iter.args:
                val = self.evaluator.evaluate(arg)
                if val is None:
                    raise ValueError("Range arguments must be classical values")
                range_args.append(val)

            # Create range based on evaluated arguments
            if len(range_args) == 1:
                loop_range = range(range_args[0])
            elif len(range_args) == 2:
                loop_range = range(range_args[0], range_args[1])
            elif len(range_args) == 3:
                loop_range = range(range_args[0], range_args[1], range_args[2])
            else:
                raise ValueError("Invalid number of arguments to range()")

            # Unroll the loop
            unrolled_body = []
            for i in loop_range:
                # For each iteration, make a copy of the body
                for stmt in node.body:
                    # Clone statement to avoid modifying original
                    stmt_copy = copy.deepcopy(stmt)

                    # Replace loop variable with its value
                    loop_var_replacer = LoopVariableReplacer(node.target.id, i)
                    transformed_stmt = loop_var_replacer.visit(stmt_copy)

                    # Process the transformed statement
                    processed_stmt = self.visit(transformed_stmt)

                    # Add to unrolled body
                    if isinstance(processed_stmt, list):
                        unrolled_body.extend(processed_stmt)
                    else:
                        unrolled_body.append(processed_stmt)

            return unrolled_body

        # Only classical loops are supported
        raise ValueError("Only for loops with classical range are supported")

    def visit_BinOp(self, node):
        """Process binary operations."""
        # Try to evaluate as classical expression
        result = self.evaluator.evaluate(node)

        if result is not None:
            # This is a purely classical operation - replace with constant
            return ast.Constant(value=result)

        # Process left and right sides
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        return node

    def visit_Subscript(self, node):
        """Handle array access operations."""
        if isinstance(node.value, ast.Name):
            var_name = node.value.id

            # If this is a classical array being indexed
            if var_name in self.classical_inputs:
                # Try to evaluate the index
                index_value = self.evaluator.evaluate(node.slice)

                if index_value is not None:
                    # Get the value from the classical array
                    array_value = self.classical_inputs[var_name]

                    try:
                        # Replace with the constant value
                        return ast.Constant(value=array_value[index_value])
                    except (IndexError, TypeError):
                        pass  # Fall through to default handling

        # Regular processing
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node

    def visit_Name(self, node):
        """Process variable references."""
        if isinstance(node.ctx, ast.Load):
            # Check if this is a reference to a classical variable
            if node.id in self.classical_inputs:
                return ast.Constant(value=self.classical_inputs[node.id])

        return node


class LoopVariableReplacer(ast.NodeTransformer):
    """
    Replaces references to loop variables with their values.
    """

    def __init__(self, loop_var: str, value: Any):
        self.loop_var = loop_var
        self.value = value
        super().__init__()

    def visit_Name(self, node):
        """Replace loop variable references with constant values."""
        if node.id == self.loop_var and isinstance(node.ctx, ast.Load):
            return ast.Constant(value=self.value)
        return node


class FunctionCallHandler:
    """Simple handler for quantum function calls."""

    def __init__(self, globals_dict: Dict):
        """
        Initialize with the global namespace to find functions.

        Args:
            globals_dict: Global namespace dictionary
        """
        self.globals = globals_dict

    def process_call(
        self, call_node: ast.Call, evaluator, used_names: set
    ) -> Optional[ast.AST]:
        """
        Process a function call and transform it if it's a quantum function.

        Args:
            call_node: The function call AST node
            evaluator: ClassicalExpressionEvaluator instance
            used_names: Set of variable names in use

        Returns:
            A new AST node if the function was transformed, None otherwise
        """
        if not isinstance(call_node.func, ast.Name):
            return None

        func_name = call_node.func.id
        if func_name not in self.globals or not callable(self.globals[func_name]):
            return None

        func = self.globals[func_name]
        if not hasattr(func, "_quantum_params"):
            return None

        # This is a quantum function call - we need to transform it

        # 1. Evaluate the arguments
        call_args = {}
        for i, arg in enumerate(call_node.args):
            # Try to evaluate if it's a classical expression
            value = evaluator.evaluate(arg)
            if value is not None:
                call_args[f"arg_{i}"] = value

        # 2. Get the quantum parameters for this function
        quantum_params = func._quantum_params

        # 3. Process any lambda functions in the quantum params
        processed_params = {}
        for name, param_spec in quantum_params.items():
            if callable(param_spec) and not hasattr(param_spec, "_quantum_params"):
                # This is a lambda - evaluate it
                try:
                    sig = inspect.signature(param_spec)
                    param_names = list(sig.parameters.keys())
                    args = [call_args.get(f"arg_{i}") for i in range(len(param_names))]
                    processed_params[name] = param_spec(*args)
                except Exception as e:
                    raise ValueError(
                        f"Error evaluating lambda in {func_name}: {str(e)}"
                    ) from e
            else:
                processed_params[name] = param_spec

        _, transformed_func = transform_function(func, call_args, processed_params)

        # 5. Generate a unique variable name for the result
        result_var = f"inline_{func_name}_{len(used_names)}_result"
        used_names.add(result_var)

        # 6. Return a reference to the result variable
        return ast.Name(id=result_var, ctx=ast.Load())
