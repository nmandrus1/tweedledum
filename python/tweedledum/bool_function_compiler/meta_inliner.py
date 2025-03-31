import ast
import copy


class TweedledumMetaInliner(ast.NodeTransformer):
    """
    AST transformer for inlining generator function calls.

    This version properly shares classical input values with the generator functions.
    """

    def __init__(self, generator_funcs, classical_inputs=None):
        self.generator_funcs = generator_funcs or {}
        self.classical_inputs = classical_inputs or {}
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
                # Get the generator function
                generator_func = self.generator_funcs[node.value.func.id]

                # Create a context for the generator function with access to classical inputs
                # This is the key part: we're directly giving access to the classical inputs
                generator_func.classical_inputs = self.classical_inputs

                # Call the generator with the AST node arguments
                statements = generator_func(*node.value.args)

                # If we got a valid list of statements, return them
                if statements and isinstance(statements, list):
                    return statements

                # Otherwise return an empty list (remove the expression)
                return []

            except Exception as e:
                # Provide helpful error message with context
                raise ValueError(
                    f"Error in generator '{node.value.func.id}': {str(e)}\n"
                    f"Available classical inputs: {list(self.classical_inputs.keys())}"
                ) from e

        # Not a generator call, continue normal processing
        return self.generic_visit(node)


# Function to transform code with meta-functions
def transform_function_with_meta(
    func, classical_inputs, quantum_params, generator_funcs
):
    """
    Transform a function with meta-function calls and classical/quantum separation.

    Args:
        func: The function to transform
        classical_inputs: Dictionary of classical parameter values
        quantum_params: Dictionary of quantum parameter specifications
        generator_funcs: Dictionary of generator functions

    Returns:
        Transformed function with quantum parameters
    """
    import ast
    import inspect

    from .transformer import QuantumCircuitTransformer

    # Extract source code
    source = inspect.getsource(func)

    # Parse into AST
    tree = ast.parse(source.strip())

    # First apply meta-function inlining with access to classical inputs
    meta_inliner = TweedledumMetaInliner(
        generator_funcs=generator_funcs, classical_inputs=classical_inputs
    )

    # Make a deep copy to avoid modifying the original AST
    tree_copy = copy.deepcopy(tree)

    # Apply the meta-inliner
    inlined_tree = meta_inliner.visit(tree_copy)

    # Fix line numbers and context references
    ast.fix_missing_locations(inlined_tree)

    # Now apply classical/quantum separation
    transformer = QuantumCircuitTransformer(
        classical_inputs=classical_inputs,
        quantum_params=quantum_params,
        globals_dict={},
        used_names=set(),
    )

    # Transform the AST after inlining
    quantum_tree = transformer.transform(inlined_tree)

    # Generate new function code
    new_source = ast.unparse(quantum_tree)
    print(f"\nGenerated quantum-only function:\n{new_source}")

    # Compile the transformed function
    namespace = {
        "BitVec": func.__globals__["BitVec"],
        # Make sure any other necessary functions/classes are available
    }

    # Execute code to define function
    exec(new_source, func.__globals__, namespace)

    # Extract the function name
    func_name = quantum_tree.body[0].name

    # Return the specialized function
    return new_source, namespace[func_name]


# Update the transform_function in transformer.py to use this pipeline
def transform_function(func, classical_inputs, quantum_params):
    """
    Main transformation function that properly handles meta-functions.

    Args:
        func: The function to transform
        classical_inputs: Dictionary of classical parameter values
        quantum_params: Dictionary of quantum parameter specifications

    Returns:
        Transformed quantum function
    """
    # Import meta-functions from your registry
    from .meta_fns import _global_generators

    # Get registered generator functions
    # generator_funcs = get_all_generators()

    # Use the enhanced pipeline
    return transform_function_with_meta(
        func, classical_inputs, quantum_params, _global_generators
    )
