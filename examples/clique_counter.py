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
    print(f"SOURCE: \n{src}")
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

# -------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# -------------------------------------------------------------------------------
from typing import Union, List
from os.path import isfile
import inspect
import math
import types

from tweedledum.bool_function_compiler.function_parser import FunctionParser
from tweedledum.classical import TruthTable, create_from_binary_string
from tweedledum import classical


class BoolFunctionTester(object):
    """Class to represent a Boolean function

    Formally, a Boolean function is a mapping :math:`f : {0, 1}^n \to {0, 1}^m`,
    where :math:`n`(:math:`m`) is the number of inputs(outputs).  There are
    many ways to represent/specify a Boolean function.  Here, we use two:

    Truth tables:  They are are an explicit and _not compact_ function
        representation.  Basically a truth table is an exhaustive mapping from
        input binary bit-strings of length :math:`n` to corresponding output
        bit-strings of length :math:`m`. Hence they do not scale well. They are,
        however, tremendously useful to represent and manipulate small functions.

    Logic network (Xor-And graph, XAG): A logic network is modeled by a directed
        acyclic graph where nodes represent primary inputs and outputs, as well
        as local functions.  The nodes representing local function as called
        gates.  In our case, we limit our local functions to be either a 2-input
        AND or a 2-input XOR---a structure known as XAG.  Therefore, a XAG is a
        2-regular non-homogeneous logic network.

    Under the hood both representations are implemented in C++.
    """

    def __init__(self, src):
        parsed_function = FunctionParser(src.strip())
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

    def num_inputs(self):
        return len(self._parameters_signature)

    def num_outputs(self):
        return len(self._return_signature)

    def num_input_bits(self):
        return self._num_input_bits

    def num_output_bits(self):
        return self._num_output_bits

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

    def logic_network(self):
        return self._logic_network

    def truth_table(self, output_bit: int):
        if not isinstance(output_bit, int):
            raise TypeError("Parameter output must be an integer")
        if self._truth_table == None:
            self.simulate_all()
        return self._truth_table[output_bit]