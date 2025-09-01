# bool_function_compiler/quantum_circuit_function.py
import inspect
import types  # For types.LambdaType
from typing import Tuple, List, Dict, Any

from .._tweedledum import classical
from .._tweedledum.classical import TruthTable, create_from_binary_string, optimize
from .._tweedledum.passes import linear_resynth, parity_decomp
from .._tweedledum.synthesis import xag_cleanup2, xag_synth, pkrm_synth
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

    def _format_simulation_result(self, sim_result):
        i = 0
        result = list()
        for type_, size in self._return_signature:
            tmp = sim_result[i : i + size]
            result.append(type_(size, tmp[::-1]))
            i += size
        if len(result) == 1:
            return result[0]
        return tuple(result)

    def simulate(self, *argv):
        if len(argv) != self.num_inputs():
            raise RuntimeError(
                f"The function requires {self.num_inputs()}. "
                f"It's signature is: {self._parameters_signature}"
            )
        input_str = str()
        for i, arg in enumerate(argv):
            arg_type = (type(arg), len(arg))
            if arg_type != self._parameters_signature[i]:
                raise TypeError(
                    f"Wrong argument type. Argument {i} "
                    f"expected: {self._parameters_signature[i]}, "
                    f"got: {arg_type}"
                )
            arg_str = str(arg)
            input_str += arg_str[::-1]

        # If the truth table was already computed, we just need to look for the
        # result of this particular input
        if self._truth_table != None:
            position = int(input_str[::-1], base=2)
            sim_result = "".join([str(int(tt[position])) for tt in self._truth_table])
        else:
            input_vector = [bool(int(i)) for i in input_str]
            sim_result = classical.simulate(self._logic_network, input_vector)
            sim_result = "".join([str(int(i)) for i in sim_result])

        return self._format_simulation_result(sim_result)

    def simulate_all(self):
        if self._truth_table == None:
            self._truth_table = classical.simulate(self._logic_network)

        result = list()
        for position in range(2 ** self._logic_network.num_pis()):
            sim_result = "".join([str(int(tt[position])) for tt in self._truth_table])
            result.append(self._format_simulation_result(sim_result))

        return result

    def truth_table(self, output_bit: int):
        if not isinstance(output_bit, int):
            raise TypeError("Parameter output must be an integer")
        if self._truth_table == None:
            self.simulate_all()
        return self._truth_table[output_bit]

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

    def _optimize_logic_network(self):
        xag = self._logic_network
        xag = xag_cleanup2(xag)
        optimize(xag)
        self._logic_network = xag_cleanup2(xag)

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
        if output_xag_dot:
            xag_export_dot(self._logic_network, xag_dot_unoptimized_name)

        if optimize_xag:
            self._optimize_logic_network()

        # write optimized xag to DOT format
        if output_xag_dot:
            xag_export_dot(self._logic_network, xag_dot_optimized_name)

        # there is a small chance that tweedledum utterly fails
        # to synthesize a graph.
        try:
            circ = xag_synth(self._logic_network)
        except Exception as e:
            print(
                f"ERROR: {e}, tweedledum failed to synthesize a circuit, returning None"
            )
            return None

        # Circuit Optimization Passes
        if opt_parity_decomp:
            circ = parity_decomp(circ)

        if opt_linear_resynth:
            circ = linear_resynth(circ)

        return to_qiskit(circ, "gatelist")

    def truth_table_synthesis(self):
        if self._truth_table is None:
            self.simulate_all()
        circ = pkrm_synth(self._truth_table[0])
        return to_qiskit(circ, "gatelist")

    def to_3sat(self) -> Tuple[List[List[int]], Dict[str, int], float]:
        """
        Convert the quantum circuit function to 3-SAT representation.

        This method takes the already transformed (classical values substituted,
        loops unrolled) function and converts it to a 3-SAT problem.

        Returns:
            - clauses (List[List[int]]): 3-SAT clauses where the sign of integers
                                         indicates negation (negative = NOT)
            - var_mapping (Dict[str, int]): Mapping from variable names to integer
                                            identifiers used in the clauses
            - clause_ratio (float): Clause-to-variable ratio as a complexity measure
                                   (higher ratio typically means harder problem)

        Example:
            >>> qcf = QuantumCircuitFunction(my_bool_func, n=4)
            >>> clauses, var_map, ratio = qcf.to_3sat()
            >>> print(f"Generated {len(clauses)} clauses with ratio {ratio:.2f}")
            >>> print(f"Variable mapping: {var_map}")
        """
        # Import the CNF converter functionality
        from .cnf_converter import extract_3sat_from_ast, calculate_clause_ratio

        # Parse the transformed source to get the AST
        import ast

        transformed_ast = ast.parse(self.transformed_source.strip())

        # Extract 3-SAT representation
        clauses, var_mapping, output_var = extract_3sat_from_ast(transformed_ast)

        # Calculate complexity measure
        if clauses:
            num_vars = max(max(abs(lit) for lit in clause) for clause in clauses)
        else:
            num_vars = 0

        clause_ratio = calculate_clause_ratio(clauses, num_vars)

        # Log the results
        import logging

        logger = logging.getLogger(
            "tweedledum.bool_function_compiler.quantum_circuit_function"
        )
        logger.info(f"3-SAT conversion for {self.original_function_object.__name__}:")
        logger.info(f"  - Number of clauses: {len(clauses)}")
        logger.info(f"  - Number of variables: {num_vars}")
        logger.info(f"  - Clause-to-variable ratio: {clause_ratio:.2f}")
        logger.info(f"  - Output variable: {output_var}")

        return clauses, var_mapping, clause_ratio

    def get_3sat_dimacs(self) -> str:
        """
        Get the 3-SAT representation in DIMACS CNF format.

        DIMACS is the standard format for SAT problems, making it easy to
        use with SAT solvers and other tools.

        Returns:
            str: DIMACS CNF format string

        Example:
            >>> qcf = QuantumCircuitFunction(my_bool_func, n=4)
            >>> dimacs = qcf.get_3sat_dimacs()
            >>> with open("problem.cnf", "w") as f:
            ...     f.write(dimacs)
        """
        clauses, var_mapping, clause_ratio = self.to_3sat()

        # Find the maximum variable index
        if clauses:
            num_vars = max(max(abs(lit) for lit in clause) for clause in clauses)
        else:
            num_vars = 0

        # Build DIMACS format string
        lines = []

        # Header comments
        lines.append(
            f"c Generated from function: {self.original_function_object.__name__}"
        )
        lines.append(f"c Clause-to-variable ratio: {clause_ratio:.2f}")
        lines.append(f"c Variable mapping:")
        for var_name, var_id in sorted(var_mapping.items()):
            lines.append(f"c   {var_name} -> {var_id}")
        lines.append("c")

        # Problem line
        lines.append(f"p cnf {num_vars} {len(clauses)}")

        # Clauses
        for clause in clauses:
            clause_str = " ".join(str(lit) for lit in clause) + " 0"
            lines.append(clause_str)

        return "\n".join(lines)
