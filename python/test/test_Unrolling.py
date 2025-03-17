import io
from contextlib import redirect_stdout

import pytest
from tweedledum.bool_function_compiler.loop_unroller import UnrollingFunctionParser


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


def unroll(src):
    return UnrollingFunctionParser(src).unrolled_source


#################################
# TESTS
#################################


class TestLoopUnroller:
    def test_clique_counter(self):
        """Test unrolling clique counter"""
        code = """

def clique_counter_small(edges: BitVec(16), vertices: BitVec(4)) -> BitVec(1):
    s = BitVec(1, 1)  # Start with True

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            # Make sure our final result is explicitly BitVec(1)
            s = s & ~((vertices[i] & vertices[j]) & ~edges[i * len(vertices) + j])

    # Check that at most n/2 vertices are selected
    # This is equivalent to: for any subset of (n/2+1) vertices,
    # at least one of them must NOT be selected

    # For n=4 (when unrolled):
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            for k in range(j+1, len(vertices)):
                # No three vertices can all be selected (for n=4)
                s = s & ~(vertices[i] & vertices[j] & vertices[k])

    # Since we can't do arithmetic, we must write this line by hand
    return s
        """

        expected = """
def clique_counter_small(edges: BitVec(16), vertices: BitVec(4)) -> BitVec(1):
    s = BitVec(1, 1)
    s = (s & (~ ((vertices[0] & vertices[1]) & (~ edges[1]))))
    s = (s & (~ ((vertices[0] & vertices[2]) & (~ edges[2]))))
    s = (s & (~ ((vertices[0] & vertices[3]) & (~ edges[3]))))
    s = (s & (~ ((vertices[1] & vertices[2]) & (~ edges[6]))))
    s = (s & (~ ((vertices[1] & vertices[3]) & (~ edges[7]))))
    s = (s & (~ ((vertices[2] & vertices[3]) & (~ edges[11]))))
    s = (s & (~ ((vertices[0] & vertices[1]) & vertices[2])))
    s = (s & (~ ((vertices[0] & vertices[1]) & vertices[3])))
    s = (s & (~ ((vertices[0] & vertices[2]) & vertices[3])))
    s = (s & (~ ((vertices[1] & vertices[2]) & vertices[3])))
    return s
        """
        unrolled = unroll(code)
        assert normalize_code(unrolled) == normalize_code(expected)


#     def test_simple_loop(self):
#         """Test unrolling a simple for loop"""
#         code = """
# def simple_loop() -> BitVec(1):
#     for i in range(3):
#         print('Hello World!')
#     return BitVec(1, 0)
# """
#         unrolled = unroll(code)
#         expected = """
# def simple_loop() -> BitVec(1):
#     print('Hello World!')
#     print('Hello World!')
#     print('Hello World!')
#     return BitVec(1, 0)
# """
#         assert normalize_code(unrolled) == normalize_code(expected)


# For Tweedledum integration testing (if available)
# @pytest.mark.skipif(True, reason="Tweedledum integration not available")
# class TestTweedledumIntegration:
#     def test_bitvec_compatibility(self):
#         """Test that unrolled code works with BitVec operations"""
#         # This is a placeholder for future integration tests
#         pass


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
