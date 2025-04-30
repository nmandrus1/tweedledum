import ast
import copy
import inspect

from .transformer import QuantumCircuitTransformer
from .loop_unroller import ClassicalLoopUnroller
from .bitvec import BitVec


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
                statements = generator_func(
                    *node.value.args, classical_inputs=self.classical_inputs
                )

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
    Transform a function in 3 passes:
    1) Inline meta-funcs,
    2) Unroll classical loops,
    3) Separate classical/quantum logic.

    Args:
        func: The function to transform
        classical_inputs: Dictionary of classical parameter values
        quantum_params: Dictionary of quantum parameter specifications
        generator_funcs: Dictionary of generator functions

    Returns:
        Transformed function with quantum parameters
    """
    # --- Stage 0: Preparation ---
    source = inspect.getsource(func)
    tree = ast.parse(source.strip())
    tree_copy = copy.deepcopy(tree)  # Work on a copy

    # --- Stage 1: Meta-Function Inlining ---
    meta_inliner = TweedledumMetaInliner(
        generator_funcs=generator_funcs, classical_inputs=classical_inputs
    )
    inlined_tree = meta_inliner.visit(tree_copy)
    ast.fix_missing_locations(inlined_tree)

    # --- Stage 2: Classical Loop Unrolling --- # <-- NEW STAGE
    loop_unroller = ClassicalLoopUnroller(classical_inputs=classical_inputs)
    unrolled_tree = loop_unroller.visit(inlined_tree)  # Visit the *inlined* tree
    ast.fix_missing_locations(unrolled_tree)

    # --- Stage 3: Classical/Quantum Separation ---
    # This transformer now receives an AST with loops already unrolled
    transformer = QuantumCircuitTransformer(
        classical_inputs=classical_inputs,
        quantum_params=quantum_params,
        globals_dict={},  # Pass necessary globals if needed, e.g., for evaluator
        used_names=set(),
    )
    # Pass the *unrolled* tree to the final transformer
    quantum_tree = transformer.transform(unrolled_tree)  # Use unrolled_tree

    # --- Stage 4: Code Generation & Compilation ---
    new_source = ast.unparse(quantum_tree)

    namespace = {
        "BitVec": BitVec,  # Use directly imported BitVec
        # Add any other necessary items here
    }
    # Prepare globals for exec carefully
    exec_globals = {}  # Start clean? Or copy select items from func.__globals__?
    # Using func.__globals__ can be risky.
    # Might need specific imports like math if used.
    exec(new_source, exec_globals, namespace)  # Provide globals if needed

    func_name = quantum_tree.body[0].name
    # Return the source string AND the compiled function object
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
