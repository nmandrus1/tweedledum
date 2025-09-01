import ast
import copy
from typing import Any, Dict, List, Optional, Union

from .ast_lib import (
    _resolve_classical_value,
    _make_and,
    _make_load_name,
    _make_store_name,
    _make_constant,
    _make_subscript,
    _make_constant,
    _make_assignment,
    _make_store_name,
    _make_binary_op,
    _make_or,
    _make_binary_op,
    _make_and,
    _make_binary_op,
)


import ast
from typing import List, Dict, Any, Union


def generate_at_least_k_counter(vertices, n, k, classical_inputs=None):
    """
    Meta function that generates "at least k" counter expressions.
    Simple fix: just use n_val and k_val consistently.
    """
    statements = []

    # Extract actual values from AST nodes or classical_inputs
    if isinstance(n, ast.Name):
        n_val = classical_inputs[n.id]
    elif isinstance(n, ast.Constant):
        n_val = n.value
    else:
        n_val = n

    if isinstance(k, ast.Name):
        k_val = classical_inputs[k.id]
    elif isinstance(k, ast.Constant):
        k_val = k.value
    else:
        k_val = k

    # Base case: s_0_0 = True
    statements.append(ast.parse("s_0_0 = BitVec(1, 1)").body[0])

    # Base cases: s_0_j = False for j > 0
    for j in range(1, k_val + 1):
        statements.append(ast.parse(f"s_0_{j} = BitVec(1, 0)").body[0])

    # Generate counter
    for i in range(1, n_val + 1):
        statements.append(ast.parse(f"s_{i}_0 = BitVec(1, 1)").body[0])

        # Create all s_i_j up to k to ensure all references exist
        for j in range(1, k_val + 1):
            if j <= i:
                # Normal case: s[i][j] = s[i-1][j] | (vertices[i-1] & s[i-1][j-1])
                expr = f"s_{i}_{j} = s_{i - 1}_{j} | (vertices[{i - 1}] & s_{i - 1}_{j - 1})"
                statements.append(ast.parse(expr).body[0])
            else:
                # Impossible case: can't have more than i items from i inputs
                statements.append(ast.parse(f"s_{i}_{j} = BitVec(1, 0)").body[0])

    # Final result - THIS IS THE FIX: use n_val and k_val, not n and k!
    statements.append(ast.parse(f"at_least_k = s_{n_val}_{k_val}").body[0])

    return statements


def generate_exactly_k_counter(
    input_vec_node: ast.Name,
    n_node: Union[ast.Name, ast.Constant],
    k_node: Union[ast.Name, ast.Constant],
    classical_inputs: Optional[Dict[str, Any]] = None,
) -> List[ast.Assign]:
    """
    Generates boolean expressions for "exactly k of n bits are true".
    Builds both "at least k" and "at most k" constraints.
    """
    n = _resolve_classical_value(n_node, "n", classical_inputs)
    k = _resolve_classical_value(k_node, "k", classical_inputs)
    vec_name = input_vec_node.id

    statements = []

    # Track both lower and upper bounds
    # s[i][j] = "at least j of first i inputs are true"
    # t[i][j] = "at most j of first i inputs are true"

    statements.append(_make_assignment("s_0_0", _make_constant(True)))
    statements.append(_make_assignment("t_0_0", _make_constant(True)))

    for i in range(1, n + 1):
        statements.append(_make_assignment(f"s_{i}_0", _make_constant(True)))

        for j in range(0, min(i + 1, k + 2)):
            input_bit = _make_subscript(_make_load_name(vec_name), i - 1)

            if j > 0 and j <= k:
                # At least j: s[i][j] = s[i-1][j] OR (input[i-1] AND s[i-1][j-1])
                prev_j = _make_load_name(f"s_{i - 1}_{j}")
                prev_j_minus_1 = _make_load_name(f"s_{i - 1}_{j - 1}")
                and_term = _make_and(input_bit, prev_j_minus_1)
                statements.append(
                    _make_assignment(f"s_{i}_{j}", _make_or(prev_j, and_term))
                )

            if j <= k:
                # At most j: t[i][j] = t[i-1][j] AND (NOT input[i-1] OR t[i-1][j-1])
                if j == 0:
                    # t[i][0] = t[i-1][0] AND NOT input[i-1]
                    prev_t = _make_load_name(f"t_{i - 1}_0")
                    not_input = _make_unary_op(input_bit, ast.Invert())
                    statements.append(
                        _make_assignment(f"t_{i}_0", _make_and(prev_t, not_input))
                    )
                else:
                    prev_t = _make_load_name(f"t_{i - 1}_{j}")
                    prev_t_minus_1 = _make_load_name(f"t_{i - 1}_{j - 1}")
                    not_input = _make_unary_op(input_bit, ast.Invert())
                    or_term = _make_or(not_input, prev_t_minus_1)
                    statements.append(
                        _make_assignment(f"t_{i}_{j}", _make_and(prev_t, or_term))
                    )

    # exactly k = (at least k) AND (at most k)
    at_least = _make_load_name(f"s_{n}_{k}")
    at_most = _make_load_name(f"t_{n}_{k}")
    statements.append(_make_assignment("exactly_k", _make_and(at_least, at_most)))

    return statements


def _make_unary_op(operand: ast.expr, op: ast.unaryop) -> ast.UnaryOp:
    """Creates an AST UnaryOp node."""
    operand_load = copy.deepcopy(operand)
    if hasattr(operand_load, "ctx"):
        operand_load.ctx = ast.Load()
    return ast.UnaryOp(op=op, operand=operand_load)
