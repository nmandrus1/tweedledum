import functools
import inspect
import types


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
            from .meta_inliner import transform_function

            # Get function signature
            sig = inspect.signature(func)

            # Identify classical parameters from args/kwargs
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            classical_inputs = {}

            # Extract classical inputs
            quantum_param_names = set(param_specs.keys())

            for name, value in bound_args.arguments.items():
                if name not in quantum_param_names:
                    classical_inputs[name] = value

            # Process quantum parameters, evaluating lambdas if necessary
            processed_quantum_params = {}
            for name, spec in param_specs.items():
                # Check if this is a lambda function
                if isinstance(spec, types.LambdaType):
                    # Get the parameter names of the lambda
                    lambda_sig = inspect.signature(spec)
                    lambda_param_names = list(lambda_sig.parameters.keys())

                    # Get the corresponding values from classical_inputs
                    lambda_args = []
                    for param_name in lambda_param_names:
                        if param_name not in classical_inputs:
                            raise ValueError(
                                f"Lambda parameter '{param_name}' not found in classical inputs. "
                                f"Available inputs: {list(classical_inputs.keys())}"
                            )
                        lambda_args.append(classical_inputs[param_name])

                    # Evaluate the lambda with the correct arguments
                    try:
                        processed_quantum_params[name] = spec(*lambda_args)
                    except Exception as e:
                        raise ValueError(
                            f"Error evaluating lambda for parameter '{name}': {str(e)}"
                        ) from e
                else:
                    # Not a lambda, use as is
                    processed_quantum_params[name] = spec

            # Generate specialized quantum-only function
            specialized_func = transform_function(
                func, classical_inputs, processed_quantum_params
            )

            return specialized_func

        return wrapper

    return decorator
