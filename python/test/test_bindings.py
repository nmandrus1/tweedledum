# ------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# ------------------------------------------------------------------------------
"""Smoke tests for the C++ bindings most used by downstream research code:
synthesis methods, circuit passes and the Qiskit converters."""
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from tweedledum.classical import TruthTable, create_from_binary_string
from tweedledum.passes import linear_resynth, parity_decomp
from tweedledum.qiskit import from_qiskit, to_qiskit
from tweedledum.synthesis import pkrm_synth, pprm_synth, transform_synth
from tweedledum.utils import xag_export_dot
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input


# Note: functions compiled by QuantumCircuitFunction must be defined at module
# level (nested definitions are not dedented by the compiler).
@circuit_input(x=lambda n: BitVec(n))
def majority3(n: int) -> BitVec(1):
    return (x[0] & x[1]) | (x[0] & x[2]) | (x[1] & x[2])


def run_on_basis_state(circuit, x):
    """Return the basis state index |y> produced by ``circuit`` on input |x>."""
    qc = QuantumCircuit(circuit.num_qubits)
    for i in range(circuit.num_qubits):
        if (x >> i) & 1:
            qc.x(i)
    qc.compose(circuit, inplace=True)
    probs = Statevector.from_instruction(qc).probabilities()
    y = int(probs.argmax())
    assert probs[y] == pytest.approx(1.0)
    return y


def make_truth_table(bits: str) -> TruthTable:
    """``bits`` is written MSB first, i.e. ``bits[-1]`` is f(0)."""
    num_vars = (len(bits) - 1).bit_length()
    tt = TruthTable(num_vars)
    create_from_binary_string(tt, bits)
    return tt


@pytest.mark.parametrize("synth", [pkrm_synth, pprm_synth], ids=["pkrm", "pprm"])
@pytest.mark.parametrize("bits", ["10000110", "01101001", "11111110", "0110"])
def test_esop_synthesis_computes_function(synth, bits):
    tt = make_truth_table(bits)
    n = tt.num_vars()
    circuit = to_qiskit(synth(tt), "gatelist")
    assert circuit.num_qubits == n + 1
    for x in range(2**n):
        expected = int(bits[::-1][x])
        y = run_on_basis_state(circuit, x)
        assert y & ((1 << n) - 1) == x, "inputs must be preserved"
        assert (y >> n) & 1 == expected, f"f({x:0{n}b})"


# decomp_synth is not included: it emits high-level `ext.truth_table` operators
# that to_qiskit cannot convert (it is covered by the C++ test suite instead).
@pytest.mark.parametrize("synth", [transform_synth], ids=["transform"])
@pytest.mark.parametrize(
    "perm", [[0, 2, 3, 5, 7, 1, 4, 6], [1, 0, 3, 2], [7, 6, 5, 4, 3, 2, 1, 0]]
)
def test_permutation_synthesis(synth, perm):
    circuit = to_qiskit(synth(perm), "gatelist")
    for x, px in enumerate(perm):
        assert run_on_basis_state(circuit, x) == px


def sample_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.cx(1, 2)
    qc.s(2)
    qc.cx(0, 2)
    qc.rz(0.3, 2)
    qc.ccx(0, 1, 2)
    qc.swap(0, 2)
    return qc


@pytest.mark.parametrize("circuit_type", ["gatelist", "dag"])
def test_qiskit_round_trip_preserves_unitary(circuit_type):
    qc = sample_circuit()
    td_circuit = from_qiskit(qc)
    assert td_circuit.num_qubits() == qc.num_qubits
    back = to_qiskit(td_circuit, circuit_type)
    if circuit_type == "dag":
        from qiskit.converters import dag_to_circuit

        back = dag_to_circuit(back)
    assert Operator(back).equiv(Operator(qc))


@pytest.mark.parametrize(
    "passes",
    [[parity_decomp], [linear_resynth], [parity_decomp, linear_resynth]],
    ids=["parity_decomp", "linear_resynth", "both"],
)
def test_passes_preserve_unitary(passes):
    qc = sample_circuit()
    circuit = from_qiskit(qc)
    for p in passes:
        circuit = p(circuit)
    assert Operator(to_qiskit(circuit, "gatelist")).equiv(Operator(qc))


def test_xag_export_dot(tmp_path):
    qcf = QuantumCircuitFunction(majority3, 3)
    out = tmp_path / "xag.dot"
    xag_export_dot(qcf.logic_network(), str(out))
    text = out.read_text()
    assert "digraph" in text
