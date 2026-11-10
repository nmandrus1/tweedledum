# ------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# ------------------------------------------------------------------------------
"""End-to-end tests for the ``QuantumCircuitFunction`` compiler pipeline.

Each test compiles a parameterised Python function, then checks it two ways
against a brute-force Python reference:

1. classical simulation of the compiled logic network, and
2. statevector simulation of the synthesised Qiskit circuit.
"""
import itertools

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input

# ------------------------------------------------------------------------------
# Functions under test
# ------------------------------------------------------------------------------


@circuit_input(vertices=lambda n: BitVec(n))
def clique_batcher(n: int, k: int, edges) -> BitVec(1):
    s = BitVec(1, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])
    generate_batcher_sort_network(vertices, n, k)
    return s & sorted_bit_0


@circuit_input(vertices=lambda n: BitVec(n))
def clique_cardinality(n: int, k: int, edges) -> BitVec(1):
    s = BitVec(1, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])
    generate_at_least_k_counter(vertices, n, k)
    return s & at_least_k


@circuit_input(x=lambda n: BitVec(n))
def parity(n: int) -> BitVec(1):
    p = BitVec(1, 0)
    for i in range(n):
        p = p ^ x[i]
    return p


@circuit_input(x=lambda n: BitVec(n))
def all_ones(n: int) -> BitVec(1):
    s = BitVec(1, 1)
    for i in range(n):
        s = s & x[i]
    return s


CLIQUE_IMPLEMENTATIONS = [clique_batcher, clique_cardinality]

# ------------------------------------------------------------------------------
# Reference implementations and helpers
# ------------------------------------------------------------------------------


def edges_from_pairs(n, pairs):
    """Flattened adjacency matrix (upper triangle) as used by the examples."""
    edges = [0] * (n * n)
    for i, j in pairs:
        i, j = min(i, j), max(i, j)
        edges[i * n + j] = 1
    return edges


def bits(x, n):
    return [(x >> i) & 1 for i in range(n)]


def reference_clique(n, k, edges, x):
    selected = [i for i, b in enumerate(bits(x, n)) if b]
    pairwise = all(edges[i * n + j] for i, j in itertools.combinations(selected, 2))
    return int(pairwise and len(selected) >= k)


def compiled_value(qcf, x, n):
    """Classically simulate the compiled logic network on input ``x``."""
    return int(str(qcf.simulate(BitVec(n, x))))


def circuit_value(circuit, x, n):
    """Run the synthesised circuit on basis state |x> and read the output qubit.

    The synthesised circuit places the inputs on qubits 0..n-1, the output on
    qubit n, and any ancillas above that.  Ancillas must be returned to |0>.
    """
    qc = QuantumCircuit(circuit.num_qubits)
    for i, b in enumerate(bits(x, n)):
        if b:
            qc.x(i)
    qc.compose(circuit, inplace=True)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    outcome, p = max(probs.items(), key=lambda kv: kv[1])
    assert p == pytest.approx(1.0), f"output is not a basis state for x={x}"
    outcome = outcome[::-1]  # qiskit strings are big-endian; index by qubit
    assert [int(c) for c in outcome[:n]] == bits(x, n), "inputs were modified"
    assert set(outcome[n + 1 :]) <= {"0"}, "ancillas were not uncomputed"
    return int(outcome[n])


# ------------------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------------------

GRAPHS = [
    # (n, k, edge list)
    pytest.param(3, 2, [(0, 1)], id="n3-k2-single-edge"),
    pytest.param(3, 3, [(0, 1), (1, 2), (0, 2)], id="n3-k3-triangle"),
    pytest.param(4, 2, [], id="n4-k2-empty"),
    pytest.param(4, 3, [(0, 1), (1, 2), (0, 2), (2, 3)], id="n4-k3-triangle-plus-tail"),
    pytest.param(4, 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], id="n4-k4-complete"),
]


@pytest.mark.parametrize("func", CLIQUE_IMPLEMENTATIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,k,pairs", GRAPHS)
def test_clique_logic_network_matches_reference(func, n, k, pairs):
    edges = edges_from_pairs(n, pairs)
    qcf = QuantumCircuitFunction(func, n, k, edges)
    assert qcf.num_input_bits() == n
    assert qcf.num_output_bits() == 1
    for x in range(2**n):
        assert compiled_value(qcf, x, n) == reference_clique(n, k, edges, x), f"x={x:0{n}b}"


@pytest.mark.parametrize("func", CLIQUE_IMPLEMENTATIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,k,pairs", GRAPHS)
def test_clique_simulate_all_matches_simulate(func, n, k, pairs):
    qcf = QuantumCircuitFunction(func, n, k, edges_from_pairs(n, pairs))
    expected = [compiled_value(qcf, x, n) for x in range(2**n)]
    assert [int(str(r)) for r in qcf.simulate_all()] == expected


# Statevector simulation cost doubles with every qubit; wider circuits are only
# checked when running with --run-slow.
MAX_FAST_QUBITS = 16


@pytest.mark.parametrize("optimize_xag", [True, False], ids=["opt", "noopt"])
@pytest.mark.parametrize("func", CLIQUE_IMPLEMENTATIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,k,pairs", GRAPHS)
def test_clique_synthesized_circuit_matches_reference(
    func, n, k, pairs, optimize_xag, request
):
    edges = edges_from_pairs(n, pairs)
    qcf = QuantumCircuitFunction(func, n, k, edges)
    circuit = qcf.synthesize_quantum_circuit(optimize_xag=optimize_xag)
    assert circuit is not None, "xag_synth failed"
    if circuit.num_qubits > MAX_FAST_QUBITS and not request.config.getoption("--run-slow"):
        pytest.skip(f"{circuit.num_qubits}-qubit circuit: use --run-slow to simulate it")
    for x in range(2**n):
        assert circuit_value(circuit, x, n) == reference_clique(n, k, edges, x), f"x={x:0{n}b}"


@pytest.mark.parametrize(
    "func,reference",
    [
        (parity, lambda xs: sum(xs) % 2),
        (all_ones, lambda xs: int(all(xs))),
    ],
    ids=["parity", "all_ones"],
)
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_simple_functions_end_to_end(func, reference, n):
    qcf = QuantumCircuitFunction(func, n)
    circuit = qcf.synthesize_quantum_circuit()
    assert circuit is not None
    for x in range(2**n):
        expected = reference(bits(x, n))
        assert compiled_value(qcf, x, n) == expected
        assert circuit_value(circuit, x, n) == expected


def test_classical_loops_are_unrolled():
    n, k = 3, 2
    qcf = QuantumCircuitFunction(clique_batcher, n, k, edges_from_pairs(n, [(0, 1)]))
    source = qcf.get_transformed_source()
    assert "for " not in source
    assert "generate_batcher_sort_network" not in source
    # Only the missing edges (0,2) and (1,2) produce constraints.
    assert "vertices[0] & vertices[2]" in source
    assert "vertices[1] & vertices[2]" in source
    assert "vertices[0] & vertices[1])" not in source


def test_undecorated_function_is_rejected():
    def plain(n: int) -> BitVec(1):
        return BitVec(1, 0)

    with pytest.raises(TypeError, match="circuit_input"):
        QuantumCircuitFunction(plain, 2)


def test_simulate_checks_argument_width():
    qcf = QuantumCircuitFunction(parity, 3)
    with pytest.raises(TypeError):
        qcf.simulate(BitVec(2, 0))


def test_to_3sat_produces_cnf():
    n, k = 3, 2
    qcf = QuantumCircuitFunction(clique_batcher, n, k, edges_from_pairs(n, [(0, 1)]))
    clauses, var_map, ratio = qcf.to_3sat()
    assert clauses, "expected a non-empty CNF"
    assert all(1 <= len(c) <= 3 for c in clauses)
    assert all(isinstance(lit, int) and lit != 0 for c in clauses for lit in c)
    assert ratio > 0
    dimacs = qcf.get_3sat_dimacs()
    assert dimacs.lstrip().startswith(("p cnf", "c"))
