# test_Compiler2.py

import pytest
import ast
import textwrap  # For cleaning up multiline strings
import inspect

# Assuming your package structure allows these imports
from tweedledum.bool_function_compiler.bitvec import BitVec
from tweedledum.bool_function_compiler.decorators import circuit_input

# transform_function is the core logic that performs the transformation
# It's defined in meta_inliner.py in your current code
from tweedledum.bool_function_compiler.meta_inliner import transform_function


# Define the function to be tested within the test file for clarity
# Or import it if it's defined elsewhere
@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_batcher(n: int, k: int, edges, vertices) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            # Using flattened edge list index: index = i * n + j
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    # Use the Batcher sorter meta-function (as in the original example context)
    # Note: The meta_inliner needs access to the generator function definitions
    # This import might be needed within transform_function or its callers
    # from .meta_fns import generate_batcher_sort_network
    generate_batcher_sort_network(vertices, n, k)

    # Assume the sorter creates 'sorted_bit_0' which is 1 if popcount >= k
    return s & sorted_bit_0


# Define the specific test case parameters
test_n = 3
test_k = 2
# Edges for a 3-node graph with only edge 0-1 connected.
# Flattened adjacency matrix (upper triangle):
# - 0-1: edges[0*3 + 1] = edges[1] = 1
# - 0-2: edges[0*3 + 2] = edges[2] = 0
# - 1-2: edges[1*3 + 2] = edges[5] = 0
# Full list based on nested loops in the function:
# i=0, j=1: index 1
# i=0, j=2: index 2
# i=1, j=2: index 5
# We need a list long enough to cover max index, size n*n = 9
# Let's assume default 0, set edge[1]=1
test_edges = [0] * (test_n * test_n)
test_edges[1] = 1  # Edge 0-1 exists

# Define the expected "golden" output code string
# This must exactly match the output provided in the user query
# Use textwrap.dedent to handle indentation in multiline strings nicely
golden_code = textwrap.dedent("""\
    def parameterized_clique_counter_batcher(vertices: BitVec(3)) -> BitVec(1):
        \"\"\"Counts cliques of size 2 in a graph specified by the edge list.\"\"\"
        s = BitVec(1, 1)
        s = s & ~(vertices[0] & vertices[2])
        s = s & ~(vertices[1] & vertices[2])
        b_tmp_h_0 = vertices[0] | vertices[1]
        b_tmp_l_0 = vertices[0] & vertices[1]
        b_tmp_h_1 = b_tmp_h_0 | vertices[2]
        b_tmp_l_1 = b_tmp_h_0 & vertices[2]
        b_tmp_h_2 = b_tmp_l_0 | b_tmp_l_1
        b_tmp_l_2 = b_tmp_l_0 & b_tmp_l_1
        sorted_bit_0 = b_tmp_h_2
        return s & sorted_bit_0""")

# --- Pytest Test Function ---


def test_clique_counter_3_nodes_1_edge_golden_code():
    """
    Tests the transformed code output against a golden string for a specific
    3-node, k=2 clique counting scenario with only edge 0-1 present.
    """
    # 1. Define classical and quantum parameters for this specific test
    classical_inputs = {"n": test_n, "k": test_k, "edges": test_edges}
    # Process the lambda function in the decorator spec
    # (This logic would ideally be inside your explicit compile function)
    quantum_param_specs = parameterized_clique_counter_batcher._quantum_params
    processed_quantum_params = {}
    for name, spec in quantum_param_specs.items():
        if callable(spec):
            lambda_sig = inspect.signature(spec)
            lambda_args = [classical_inputs[p] for p in lambda_sig.parameters]
            processed_quantum_params[name] = spec(*lambda_args)
        else:
            processed_quantum_params[name] = spec

    # 2. Run the transformation pipeline
    # transform_function returns (new_source, compiled_function_object)
    # We only need the source code string for this test.

    transformed_source, _ = transform_function(
        parameterized_clique_counter_batcher, classical_inputs, processed_quantum_params
    )

    # 3. Compare the actual output with the golden string
    # Optional: Normalize whitespace or formatting if needed, but exact match is stricter.
    # For example, parsing and unparsing again can normalize formatting:
    # parsed_actual = ast.parse(transformed_source)
    # normalized_actual_source = ast.unparse(parsed_actual)
    # parsed_golden = ast.parse(golden_code)
    # normalized_golden_source = ast.unparse(parsed_golden)
    # assert normalized_actual_source == normalized_golden_source

    # Direct string comparison (more brittle):
    assert transformed_source.strip() == golden_code.strip()


# To run this test:
# 1. Save the code as test_Compiler2.py
# 2. Make sure your bool_function_compiler package is in the Python path.
# 3. Run pytest in your terminal in the directory containing the test file:
#    pytest test_Compiler2.py
