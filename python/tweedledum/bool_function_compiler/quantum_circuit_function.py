# bool_function_compiler/quantum_circuit_function.py
import inspect
import types  # For types.LambdaType
from .._tweedledum.synthesis import xag_synth, xag_cleanup
from .._tweedledum.classical import optimize
from .._tweedledum.passes import parity_decomp, linear_resynth
from .._tweedledum.utils import xag_export_dot
from ..qiskit.converters import to_qiskit

from .bitvec import BitVec
from .decorators import CIRCUIT_QUANTUM_DEF_ATTR  # Import the attribute name
from .function_parser import FunctionParser
from .meta_fns import _global_generators
from .meta_inliner import transform_function_with_meta


class QuantumCircuitFunction:
    """
    Manages the transformation of a Python function (decorated with @circuit_input)
    into a logic network suitable for quantum circuit synthesis, using provided
    runtime arguments for the function.
    """

    def __init__(self, decorated_py_func, *func_args, **func_kwargs):
        """
        Initializes and processes the Python function with its runtime arguments.

        Args:
            decorated_py_func (function): The Python function that has been
                                         decorated with @circuit_input.
            *func_args: Positional arguments for decorated_py_func.
            **func_kwargs: Keyword arguments for decorated_py_func.
        """
        if not hasattr(decorated_py_func, CIRCUIT_QUANTUM_DEF_ATTR):
            raise TypeError(
                f"Function '{decorated_py_func.__name__}' must be decorated with @circuit_input "
                "to be used with QuantumCircuitFunction."
            )

        self.original_function_object = decorated_py_func

        # Retrieve the quantum parameter definitions attached by the decorator
        quantum_definitions = getattr(decorated_py_func, CIRCUIT_QUANTUM_DEF_ATTR)

        # Bind the provided runtime arguments to the function's signature
        # inspect.unwrap is used to get to the original function if there are multiple decorators,
        # but functools.wraps should make decorated_py_func.signature work.
        try:
            sig = inspect.signature(decorated_py_func)
            bound_arguments = sig.bind(*func_args, **func_kwargs)
            bound_arguments.apply_defaults()  # Apply defaults for any missing args
        except TypeError as e:
            raise TypeError(
                f"Error binding arguments to function '{decorated_py_func.__name__}': {e}"
            ) from e

        # Populate classical_inputs for the compilation pipeline
        # These are all arguments passed to the function that are NOT quantum definitions.
        self.classical_inputs = {}
        for param_name, value in bound_arguments.arguments.items():
            if param_name not in quantum_definitions:
                self.classical_inputs[param_name] = value

        # Process quantum_definitions to resolve lambdas using these classical_inputs
        self.processed_quantum_params = {}  # This will store name -> BitVec instance
        for q_name, spec in quantum_definitions.items():
            if isinstance(spec, types.LambdaType):
                lambda_sig = inspect.signature(spec)
                lambda_param_names = list(lambda_sig.parameters.keys())

                missing_classical_params = [
                    p_name
                    for p_name in lambda_param_names
                    if p_name not in self.classical_inputs
                ]
                if missing_classical_params:
                    raise ValueError(
                        f"Lambda for quantum parameter '{q_name}' depends on classical parameters "
                        f"not found or not provided: {missing_classical_params}. "
                        f"Available classical inputs for lambda: {list(self.classical_inputs.keys())}"
                    )

                lambda_args_values = [
                    self.classical_inputs[p_name] for p_name in lambda_param_names
                ]
                try:
                    self.processed_quantum_params[q_name] = spec(*lambda_args_values)
                except Exception as e:
                    raise ValueError(
                        f"Error evaluating lambda for quantum parameter '{q_name}' "
                        f"with args {lambda_args_values}: {str(e)}"
                    ) from e
            elif isinstance(spec, BitVec):  # If spec is already a BitVec instance
                self.processed_quantum_params[q_name] = spec
            else:
                # Assuming spec is a BitVec instance or compatible
                self.processed_quantum_params[q_name] = spec

        # Call the full AST Transformation Pipeline
        self.transformed_source, self.transformed_function_obj = (
            transform_function_with_meta(
                self.original_function_object,  # Pass the callable (potentially wrapped) function
                self.classical_inputs,  # Derived classical inputs for compilation
                self.processed_quantum_params,  # Resolved quantum param objects (e.g. BitVec instances)
                _global_generators,
            )
        )

        # Parse the transformed (quantum-only) function to build the LogicNetwork
        source_to_parse = self.transformed_source.strip()
        parsed_function = FunctionParser(source_to_parse)

        self._parameters_signature = parsed_function._parameters_signature
        self._return_signature = parsed_function._return_signature
        self._logic_network = parsed_function._logic_network
        self._truth_table = None
        self._num_input_bits = self._logic_network.num_pis()
        self._num_output_bits = self._logic_network.num_pos()

    def logic_network(self):
        return self._logic_network

    def num_inputs(self):
        return len(self._parameters_signature)

    def num_outputs(self):
        return len(self._return_signature)

    def num_input_bits(self):
        return self._num_input_bits

    def num_output_bits(self):
        return self._num_output_bits

    def get_transformed_source(self):
        return self.transformed_source

    def get_transformed_function(self):
        return self.transformed_function_obj

    def synthesize_quantum_circuit(
        self,
        optimize_xag=True,
        opt_parity_decomp=True,
        opt_linear_resynth=True,
        output_xag_dot=False,
        xag_dot_unoptimized_name="initial_xag.dot",
        xag_dot_optimized_name="optimized_xag.dot",
    ):
        # generate classical function source
        xag = self._logic_network

        # XAG operations
        xag = xag_cleanup(xag)
        if output_xag_dot:
            xag_export_dot(xag, xag_dot_unoptimized_name)

        if optimize_xag:
            optimize(xag)

        # write optimized xag to DOT format
        if output_xag_dot:
            xag_export_dot(xag, xag_dot_optimized_name)

        circ = xag_synth(xag)

        # Circuit Optimization Passes
        if opt_parity_decomp:
            circ = parity_decomp(circ)

        if opt_linear_resynth:
            circ = linear_resynth(circ)

        return to_qiskit(circ, "gatelist")
