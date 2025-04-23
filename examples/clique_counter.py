from tweedledum.bool_function_compiler.decorators import circuit_input
from tweedledum.bool_function_compiler.function_parser import FunctionParser
from tweedledum.synthesis import xag_synth, xag_cleanup
from tweedledum.classical import optimize
from tweedledum.passes import parity_decomp, linear_resynth
from tweedledum import BitVec
import tweedledum as td
from qiskit import QuantumCircuit
import networkx as nx

@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_batcher(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    # generate_sorting_network(vertices, n, k)
    generate_batcher_sort_network(vertices, n, k)

    return s & sorted_bit_0


@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    generate_sorting_network(vertices, n, k)

    return s & sorted_bit_0


def oracle_from_graph(graph: nx.Graph, clique_size: int) -> QuantumCircuit:
    # get edge list
    edges = nx.to_numpy_array(graph).flatten()
    num_nodes = graph.number_of_nodes()

    # generate classical function source
    src, _ = parameterized_clique_counter_batcher(num_nodes, clique_size, edges)
    parsed_function = FunctionParser(src.strip())
    xag = parsed_function._logic_network

    # XAG operations
    xag = xag_cleanup(xag)
    optimize(xag)
    circ = xag_synth(xag)

    # Circuit Optimization Passes
    circ = parity_decomp(circ)
    circ = linear_resynth(circ)
    return td.converters.to_qiskit(circ, "gatelist")