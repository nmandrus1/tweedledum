import io
from contextlib import redirect_stdout
import astunparse
import ast

import pytest
from tweedledum.bool_function_compiler import TweedledumMetaInliner


# Test helper functions
def normalize_code(code):
    """Normalize code by removing whitespace and comments"""
    # Strip whitespace and comments for comparison
    return "".join(
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def execute_and_capture(code):
    """Execute code and capture its output"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(code)
    return buffer.getvalue()


def expand_fns(src):
    tree = ast.parse(src.strip())
    inliner = TweedledumMetaInliner()
    transformed_tree = inliner.visit(tree)
    return astunparse.unparse(transformed_tree)


#################################
# TESTS
#################################


class TestInlining:
    def test_clique_counter(self):
        """Test unrolling clique counter"""
        code = """
        # Clique Counter
def clique_counter_sorting(edges: BitVec(16), vertices: BitVec(4)) -> BitVec(1):
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            s = s & ~((vertices[i] & vertices[j]) & ~edges[i * len(vertices) + j])

    generate_sorting_network(vertices, 'sorted', 4)

    return s & sorted_bit_1 & ~sorted_bit_2
        """

        expected = """
def clique_counter_sorting(edges: BitVec(16), vertices: BitVec(4)) -> BitVec(1):
    s = BitVec(1, 1)
    for i in range(len(vertices)):
        for j in range((i + 1), len(vertices)):
            s = (s & (~ ((vertices[i] & vertices[j]) & (~ edges[((i * len(vertices)) + j)]))))
    s_1_1_high = (vertices[0] | vertices[1])
    s_1_1_low = (vertices[0] & vertices[1])
    s_2_1_high = (s_1_1_low | vertices[2])
    s_2_1_low = (s_1_1_low & vertices[2])
    s_2_2_high = (s_1_1_high | s_2_1_high)
    s_2_2_low = (s_1_1_high & s_2_1_high)
    s_3_1_high = (s_2_1_low | vertices[3])
    s_3_1_low = (s_2_1_low & vertices[3])
    s_3_2_high = (s_2_2_low | s_3_1_high)
    s_3_2_low = (s_2_2_low & s_3_1_high)
    s_3_3_high = (s_2_2_high | s_3_2_high)
    s_3_3_low = (s_2_2_high & s_3_2_high)
    sorted_bit_1 = s_3_3_low
    sorted_bit_2 = s_3_2_low
    return ((s & sorted_bit_1) & (~ sorted_bit_2))
        """
        expanded = expand_fns(code)
        assert normalize_code(expanded) == normalize_code(expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
