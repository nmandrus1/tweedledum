import functools
import inspect


class CompileTime:
    """Type annotation for compile-time parameters."""

    def __init__(self, type_):
        self.type = type_

    def __getitem__(self, type_):
        return CompileTime(type_)


def circuit_input(**param_specs):
    """
    Decorator to specify which parameters are quantum inputs.

    Args:
        **param_specs: Mapping parameter names to BitVec types or functions that return BitVec types
                      e.g. vertices=BitVec(4) or qubits=lambda n: BitVec(n)
    """

    def decorator(func):
        # Store quantum parameter specs
        func._quantum_params = param_specs

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .transformer import transform_function

            # Get function signature
            sig = inspect.signature(func)

            # Identify classical parameters from args/kwargs
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            classical_inputs = {}

            # Extract classical inputs
            param_names = list(sig.parameters.keys())
            quantum_param_names = set(param_specs.keys())

            for name, value in bound_args.arguments.items():
                if name not in quantum_param_names:
                    classical_inputs[name] = value

            # Generate specialized quantum-only function
            specialized_func = transform_function(func, classical_inputs, param_specs)

            return specialized_func

        return wrapper

    return decorator
