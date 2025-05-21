# bool_function_compiler/decorators.py
import functools

# Define a consistent attribute name for storing the quantum definitions
CIRCUIT_QUANTUM_DEF_ATTR = "_circuit_quantum_definitions"


def circuit_input(**quantum_parameter_definitions):
    """
    Decorator to mark a function for quantum circuit synthesis and store the
    definitions of its quantum parameters.

    The definitions are stored in an attribute (named by CIRCUIT_QUANTUM_DEF_ATTR)
    on the decorated function object.

    Args:
        **quantum_parameter_definitions: Keyword arguments mapping quantum parameter
            names to their specifications. Specifications can be BitVec instances
            or lambda functions that take classical parameter values and return
            BitVec instances.
            Example: @circuit_input(vars=lambda n: BitVec(n), fixed_ancilla=BitVec(2))
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This wrapper is called if the user *calls* the decorated function.
            # The primary intention for compilation is to pass the `wrapper` object
            # (which is what `func` becomes after decoration) to QuantumCircuitFunction.
            # If called directly, it should behave like the original function.
            return func(*args, **kwargs)

        # Attach the quantum parameter definitions to the wrapper object.
        # This is what QuantumCircuitFunction will access.
        setattr(wrapper, CIRCUIT_QUANTUM_DEF_ATTR, quantum_parameter_definitions)
        return wrapper

    return decorator
