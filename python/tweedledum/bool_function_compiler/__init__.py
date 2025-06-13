# -------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# -------------------------------------------------------------------------------
import logging

from tweedledum.bool_function_compiler import (
    classical_expression_evaluator as classical_expression_evaluator,
)
from tweedledum.bool_function_compiler import (
    transformer as transformer,
)
from tweedledum.bool_function_compiler import (
    variable_classifier as variable_classifier,
)
from tweedledum.ir import Circuit
from tweedledum.operators import H, X
from tweedledum.synthesis import (
    lhrs_synth,
    pkrm_synth,
    pprm_synth,
    spectrum_synth,
    xag_synth,
)

from .bitvec import BitVec as BitVec
from .bool_function import BoolFunction
from .decorators import circuit_input as circuit_input
from typing import Union, Optional, Dict

# Compiler 2.0
from .quantum_circuit_function import QuantumCircuitFunction as QuantumCircuitFunction

# Library logger name
TWEEDLEDUM_LOGGER = "tweedledum"


def setup_logging(
    level: Union[int, str] = logging.WARNING,
    module_levels: Optional[Dict[str, Union[int, str]]] = None,
    format_string: Optional[str] = None,
    enable_console: bool = True,
):
    """
    Setup logging for Tweedledum library only.

    Args:
        level: Default logging level for all Tweedledum modules
        module_levels: Dict of specific levels for individual modules, e.g.:
                      {"variable_classifier": logging.DEBUG, "transformer": logging.INFO}
        format_string: Custom format string (defaults to simple format)
        enable_console: Whether to output to console

    Examples:
        # Quiet (default)
        setup_logging()

        # Debug everything
        setup_logging(logging.DEBUG)

        # Debug specific modules only
        setup_logging(logging.WARNING, {
            "variable_classifier": logging.DEBUG,
            "transformer": logging.INFO
        })
    """
    # Get the main library logger
    tweedledum_logger = logging.getLogger(TWEEDLEDUM_LOGGER)

    # Clear any existing handlers to avoid duplicates
    tweedledum_logger.handlers.clear()

    # Set the base level
    tweedledum_logger.setLevel(level)

    # Don't propagate to root logger (this is the key to not affecting other libraries)
    tweedledum_logger.propagate = False

    if enable_console:
        # Create and configure handler
        handler = logging.StreamHandler()

        if format_string is None:
            format_string = "%(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        handler.setLevel(level)

        # Add handler to our library logger
        tweedledum_logger.addHandler(handler)

    # Set specific module levels if provided
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(
                f"{TWEEDLEDUM_LOGGER}.bool_function_compiler.{module_name}"
            )
            module_logger.setLevel(module_level)


# Convenience functions for common use cases
def enable_debug():
    """Enable debug logging for all Tweedledum modules."""
    setup_logging(logging.DEBUG)


def silence():
    """Silence all Tweedledum logging."""
    setup_logging(logging.CRITICAL + 1, enable_console=False)


setup_logging()


_METHOD_TO_CALLABLE = {
    "lhrs": lhrs_synth,
    "pprm": pprm_synth,
    "pkrm": pkrm_synth,
    "spectrum": spectrum_synth,
    "xag": xag_synth,
}


def bitflip_circuit(f: BoolFunction, method: str, config: dict() = {}):
    r"""Synthesizes a Boolean Function as a bitflip oracle.

    A bitflip oracle is a quantum operator `Bf` specified by a Boolean function
    `f` for which the effect on all computational basis states is given by

       Bf : |x>|y>|0>^a --> |x>|y + f(x)>|0>^a

    where `+` is the logical exclusive-or operator and `a >= 0` corresponds to
    the number of extra qubits used to store intermediate results for the
    computation of `f(x)`, the so-called ancillae qubits.

    (For clarity, this example only showed a single output function)

    Args:
        f(BoolFunction): Boolean function to be synthesized
        method: Synthesis method ('lhrs', 'pprm', 'pprm', 'spectrum', or 'xag')
        config: Dictionary with configuration parameters for tweedledum
    """
    synthesizer = _METHOD_TO_CALLABLE.get(method)
    if synthesizer is None:
        raise ValueError(f"Unrecognized synthesis method: {method}")
    if method in ["pprm", "pkrm", "spectrum"]:
        if f.num_outputs() > 1:
            raise ValueError("TT based methods only work for single output functions")
        if f.num_inputs() > 16:
            raise ValueError(
                "TT based methods only work for functions with at most 16 inputs"
            )
        return synthesizer(f.truth_table(output_bit=0), config)
    return synthesizer(f.logic_network(), config)


def phaseflip_circuit(f: BoolFunction, method: str, config: dict() = {}):
    r"""Synthesizes a Boolean Function as a phaseflip oracle.

    A phaseflip oracle is a quantum operator `Pf` specified by a Boolean
    function `f` for which the effect on all computational basis states is given
    by

        Bf : |x>|0>^a --> (-1)^{f(x)}|x>|0>^a

    Note that you can easily construct a phase oracle from a bit oracle by
    sandwiching the controlled X gate on the result qubit by a X and H gate.
    """
    synthesizer = _METHOD_TO_CALLABLE.get(method)
    if synthesizer is None:
        raise ValueError(f"Unrecognized synthesis method: {method}")
    if method in ["pprm", "pkrm", "spectrum"]:
        if f.num_outputs() > 1:
            raise ValueError("TT based methods only work for single output functions")
        if f.num_inputs() > 16:
            raise ValueError(
                "TT based methods only work for functions with at most 16 inputs"
            )
        if f"{method}_synth" not in config:
            config[f"{method}_synth"] = {"phase_esop": True}
        else:
            config[f"{method}_synth"]["phase_esop"] = True
        return synthesizer(f.truth_table(output_bit=0), config)

    circuit = Circuit()
    num_qubits = f.num_inputs() + f.num_outputs()
    qubits = [circuit.create_qubit() for _ in range(num_qubits)]
    cbits = list()
    for i in range(f.num_inputs(), num_qubits):
        circuit.apply_operator(X(), [qubits[i]])
        circuit.apply_operator(H(), [qubits[i]])
    synthesizer(circuit, qubits, cbits, f.logic_network(), config)
    for i in range(f.num_inputs(), num_qubits):
        circuit.apply_operator(H(), [qubits[i]])
        circuit.apply_operator(X(), [qubits[i]])
    return circuit
