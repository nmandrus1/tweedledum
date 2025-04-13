import ast
import copy
import math  # Added for ceiling and log2
from typing import Any, Dict, List, Optional, Union

import astunparse

# --- Re-include necessary helpers from your sorting_network.py ---
# (Or ensure they are importable)


def _resolve_classical_value(
    param_node: Union[ast.Name, ast.Constant],
    param_name: str,
    classical_inputs: Optional[Dict[str, Any]],
) -> int:
    # Implementation from your file
    if isinstance(param_node, ast.Constant):
        if not isinstance(param_node.value, int):
            raise ValueError(
                f"Parameter '{param_name}' must be an integer constant, is {type(param_node.value)}"
            )
        return param_node.value
    elif isinstance(param_node, ast.Name):
        param_id = param_node.id
        if classical_inputs is None:
            raise TypeError(
                f"Cannot resolve named parameter '{param_id}' without classical_inputs dictionary."
            )
        if param_id not in classical_inputs:
            raise ValueError(
                f"Unable to determine value for '{param_name}' from variable '{param_id}'. "
                f"Ensure it's present in classical_inputs."
            )
        value = classical_inputs[param_id]
        if not isinstance(value, int):
            raise ValueError(
                f"Classical input '{param_id}' for parameter '{param_name}' must be an integer."
            )
        return value
    else:
        raise ValueError(
            f"Parameter '{param_name}' must be represented by an ast.Constant or ast.Name node."
        )


def _make_load_name(var_name: str) -> ast.Name:
    """Creates an AST Name node in Load context."""
    return ast.Name(id=var_name, ctx=ast.Load())


def _make_store_name(var_name: str) -> ast.Name:
    """Creates an AST Name node in Store context."""
    return ast.Name(id=var_name, ctx=ast.Store())


def _make_constant(value: Any) -> ast.Constant:
    """Creates an AST Constant node."""
    return ast.Constant(value=value)


def _make_subscript(vec_node: ast.expr, index_val: int) -> ast.Subscript:
    """Creates an AST Subscript node vec[index]."""
    vec_load = copy.deepcopy(vec_node)
    vec_load.ctx = ast.Load()
    return ast.Subscript(
        value=vec_load,
        slice=_make_constant(value=index_val),
        ctx=ast.Load(),
    )


def _make_assignment(target_var_name: str, value_node: ast.expr) -> ast.Assign:
    """Creates an AST Assignment node target = value."""
    value_load = copy.deepcopy(value_node)
    if hasattr(value_load, "ctx"):
        value_load.ctx = ast.Load()
    assign_node = ast.Assign(
        targets=[_make_store_name(target_var_name)], value=value_load
    )
    ast.fix_missing_locations(assign_node)
    return assign_node


def _make_binary_op(
    left_node: ast.expr, right_node: ast.expr, op: ast.operator
) -> ast.BinOp:
    """Creates an AST BinOp node left op right."""
    left_load = copy.deepcopy(left_node)
    if hasattr(left_load, "ctx"):
        left_load.ctx = ast.Load()
    right_load = copy.deepcopy(right_node)
    if hasattr(right_load, "ctx"):
        right_load.ctx = ast.Load()
    bin_op_node = ast.BinOp(left=left_load, op=op, right=right_load)
    ast.fix_missing_locations(bin_op_node)
    return bin_op_node


def _make_or(left_node: ast.expr, right_node: ast.expr) -> ast.BinOp:
    """Creates an AST BitOr node."""
    return _make_binary_op(left_node, right_node, ast.BitOr())


def _make_and(left_node: ast.expr, right_node: ast.expr) -> ast.BinOp:
    """Creates an AST BitAnd node."""
    return _make_binary_op(left_node, right_node, ast.BitAnd())


# --- End Re-included Helpers ---


def generate_batcher_sort_network(
    input_vec_node: ast.Name,
    n_node: Union[ast.Name, ast.Constant],
    k_node: Union[ast.Name, ast.Constant],
    classical_inputs: Optional[Dict[str, Any]] = None,
) -> List[ast.Assign]:
    """
    Generates AST assignment statements for Batcher's odd-even sorting network.

    Sorts input bits (descending order, 1s first) and assigns the k-th bit
    (index k-1) to the variable "sorted_bit_0".

    Args:
        input_vec_node: AST Name node for the input BitVec variable (e.g., vertices).
        n_node: AST Constant or Name node representing the total number of vertices (n).
        k_node: AST Constant or Name node representing the minimum clique size (k).
        classical_inputs: Dictionary providing values for n_node/k_node if they are Names.

    Returns:
        A list of ast.Assign nodes representing the Batcher sorting network logic.
    """
    # 1. Resolve and Validate Parameters
    if not isinstance(input_vec_node, ast.Name):
        raise TypeError("Input 'input_vec_node' must be an ast.Name node.")
    input_vec_name = input_vec_node.id

    num_vertices = _resolve_classical_value(n_node, "n", classical_inputs)
    min_clique_size = _resolve_classical_value(k_node, "k", classical_inputs)

    # Basic validation (as in your original function)
    if not isinstance(num_vertices, int) or num_vertices <= 0:
        raise ValueError(
            f"Number of vertices 'n' must be a positive integer, got {num_vertices}."
        )
    if not isinstance(min_clique_size, int) or min_clique_size < 1:
        raise ValueError(
            f"Minimum clique size 'k' must be >= 1, got {min_clique_size}."
        )
    if min_clique_size > num_vertices:
        raise ValueError(
            f"Cannot find clique size {min_clique_size} with only {num_vertices} vertices."
        )

    # 2. Initialize
    statements: List[ast.Assign] = []
    vec_ast_node = _make_load_name(input_vec_name)

    # Use a list to hold the AST nodes representing the current state of each wire
    # Initialize with the input nodes
    wires = [_make_subscript(vec_ast_node, i) for i in range(num_vertices)]
    # Pad conceptually to the next power of 2 for Batcher's algorithm structure
    # Actual swaps are guarded to stay within num_vertices bounds
    n_padded = 1 << (num_vertices - 1).bit_length() if num_vertices > 0 else 0

    temp_var_counter = 0  # Counter for unique intermediate variable names

    # 3. Compare-and-Swap Helper Function (Generates AST)
    def compare_swap(idx1, idx2):
        nonlocal temp_var_counter
        # Only perform if both indices are within the original 'n' bounds
        if idx1 < num_vertices and idx2 < num_vertices:
            node1 = wires[idx1]
            node2 = wires[idx2]

            # Generate unique names for intermediate results
            # Using 'b' prefix for Batcher
            high_var_name = f"b_tmp_h_{temp_var_counter}"
            low_var_name = f"b_tmp_l_{temp_var_counter}"
            temp_var_counter += 1

            # Create AST nodes for compare-and-swap logic
            # Descending sort: max (OR) goes to lower index (idx1), min (AND) goes to higher index (idx2)
            assign_high = _make_assignment(high_var_name, _make_or(node1, node2))
            assign_low = _make_assignment(low_var_name, _make_and(node1, node2))
            statements.extend([assign_high, assign_low])

            # Update the wires list to refer to the *names* (as Load nodes) of the new results
            wires[idx1] = _make_load_name(high_var_name)
            wires[idx2] = _make_load_name(low_var_name)

    # 4. Generate Batcher Network Stages (Iterative Odd-Even Mergesort)
    p = 1
    while p < n_padded:
        k = p
        while k >= 1:
            j = k % p  # Offset within the first block
            i_start = 0
            # Iterate through elements, applying compare-swap based on stage logic
            while i_start < n_padded:
                idx1 = i_start + j
                # Only compare elements within the same larger block (size p*2)
                # This check ensures correct merging stages
                if (idx1 // (p * 2)) == ((idx1 + k) // (p * 2)):
                    compare_swap(idx1, idx1 + k)
                i_start += 1
                # Optimization: Skip intervals that don't need comparison in this sub-stage
                # If the current index 'i_start' crosses a boundary related to 'k',
                # jump 'k' steps forward.
                if (i_start & k) == k:
                    i_start += k
            k //= 2
        p *= 2

    # 5. Identify Final Sorted Output Nodes
    # The 'wires' list now holds AST nodes representing the sorted outputs (first num_vertices are valid)
    outputs_nodes = wires[:num_vertices]

    # 6. Assign the Critical Boundary Bit
    boundary_index = min_clique_size - 1

    if not (0 <= boundary_index < len(outputs_nodes)):
        # This check should be redundant if parameters are validated correctly, but good practice
        raise IndexError(
            f"Calculated boundary index {boundary_index} is out of range for "
            f"sorted list of length {len(outputs_nodes)} (n={num_vertices}, k={min_clique_size})"
        )

    target_output_node = outputs_nodes[boundary_index]
    result_var_name = "sorted_bit_0"  # Hardcoded name expected by calling function
    statements.append(_make_assignment(result_var_name, target_output_node))

    # 7. Return the list of generated assignment statements
    return statements
