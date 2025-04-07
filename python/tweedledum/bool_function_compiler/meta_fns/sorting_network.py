# import ast
# import copy


# class SortPairNode:
#     # ... (keep definition as before) ...
#     def __init__(self, high, low):
#         self.high = high
#         self.low = low


# def generate_sorting_network(
#     input_vec, output_name, n_node, k_node
# ):  # Renamed for clarity
#     """
#     Generate sorting network AST statements with intermediate variables.
#     FIXED: Assigns boundary bits to sorted_bit_0 for checking at
#            least k ones, assuming descending sort.

#     Args:
#         input_vec (ast.Name): AST node for input BitVec variable
#         output_name (ast.Constant): AST node for the output variable name
#         k_node (Union[ast.Name, ast.Constant]): AST node for the size - can be variable or constant

#     Returns:
#         list: List of AST statement nodes
#     """
#     # ... (Keep Steps 1-3: validation, helpers, network generation as before) ...
#     # Validate input types
#     if not isinstance(input_vec, ast.Name):
#         raise ValueError("Input must be a variable name")

#     if not isinstance(output_name, ast.Constant):
#         raise ValueError("Output name must be a constant string")

#     output_var_name = output_name.value

#     if isinstance(k_node, ast.Constant):
#         k_val = k_node.value  # Use k_val locally, k is often used for loop vars
#     elif isinstance(n_node, ast.Name):
#         k_name = k_node.id
#         if (
#             hasattr(generate_sorting_network, "classical_inputs")  # Check if exists
#             and isinstance(
#                 generate_sorting_network.classical_inputs, dict
#             )  # Check if dict
#             and k_name in generate_sorting_network.classical_inputs
#         ):
#             k_val = generate_sorting_network.classical_inputs[k_name]
#         else:
#             # Cannot resolve classically, this function needs the value
#             raise ValueError(
#                 f"Unable to determine size 'n' from variable {k_name} classically."
#             )
#     else:
#         raise ValueError("k must be either a constant or a variable name")

#     if k_val < 2:
#         raise ValueError(
#             "k must be >= 2 i.e. must search for a clique of size 2 or greater"
#         )

#     # Extract size value if it's a constant, otherwise use the variable directly
#     if isinstance(n_node, ast.Constant):
#         n_val = n_node.value  # Use n_val locally, n is often used for loop vars
#     elif isinstance(n_node, ast.Name):
#         n_name = n_node.id
#         if (
#             hasattr(generate_sorting_network, "classical_inputs")  # Check if exists
#             and isinstance(
#                 generate_sorting_network.classical_inputs, dict
#             )  # Check if dict
#             and n_name in generate_sorting_network.classical_inputs
#         ):
#             n_val = generate_sorting_network.classical_inputs[n_name]
#         else:
#             # Cannot resolve classically, this function needs the value
#             raise ValueError(
#                 f"Unable to determine size 'n' from variable {n_name} classically."
#             )
#     else:
#         raise ValueError("k must be either a constant or a variable name")

#     input_name = input_vec.id

#     if k_val > n_val:
#         raise ValueError(
#             "Cannot find clique with size larger than the number of vertices: k = {k_val} > n = {n_val}"
#         )

#     # Helper functions (make_subscript, make_assignment, make_or, make_and)
#     # ... (Keep definitions as before) ...
#     def make_subscript(name, index):
#         return ast.Subscript(
#             value=ast.Name(id=name, ctx=ast.Load()),
#             slice=ast.Constant(value=index),
#             ctx=ast.Load(),
#         )

#     def make_assignment(target, value):
#         # Ensure target has Store context if it's an AST node already
#         if isinstance(target, str):
#             target_node = ast.Name(id=target, ctx=ast.Store())
#         else:
#             target_node = copy.deepcopy(target)  # Avoid modifying original nodes
#             target_node.ctx = ast.Store()

#         # Ensure value has Load context
#         value_node = copy.deepcopy(value)
#         if hasattr(value_node, "ctx"):
#             value_node.ctx = ast.Load()

#         return ast.Assign(targets=[target_node], value=value_node)

#     def make_or(left, right):
#         # Ensure operands are Load context
#         if isinstance(left, str):
#             left = ast.Name(id=left, ctx=ast.Load())
#         else:
#             left = copy.deepcopy(left)
#             left.ctx = ast.Load()

#         if isinstance(right, str):
#             right = ast.Name(id=right, ctx=ast.Load())
#         else:
#             right = copy.deepcopy(right)
#             right.ctx = ast.Load()

#         return ast.BinOp(left=left, op=ast.BitOr(), right=right)

#     def make_and(left, right):
#         if isinstance(left, str):
#             left = ast.Name(id=left, ctx=ast.Load())
#         else:
#             left = copy.deepcopy(left)
#             left.ctx = ast.Load()

#         if isinstance(right, str):
#             right = ast.Name(id=right, ctx=ast.Load())
#         else:
#             right = copy.deepcopy(right)
#             right.ctx = ast.Load()

#         return ast.BinOp(left=left, op=ast.BitAnd(), right=right)

#     statements = []
#     input_vars = [make_subscript(input_name, i) for i in range(n_val)]
#     nodes = [[SortPairNode(None, None) for _ in range(n_val)] for _ in range(n_val)]

#     for i in range(n_val):
#         nodes[i][0] = SortPairNode(input_vars[i], None)  # Use the AST subscript node

#     for i in range(1, n_val):
#         for j in range(1, i + 1):
#             s_high_name = f"s_{i}_{j}_high"
#             s_low_name = f"s_{i}_{j}_low"

#             nodes[i][j] = SortPairNode(
#                 ast.Name(id=s_high_name, ctx=ast.Load()),
#                 ast.Name(id=s_low_name, ctx=ast.Load()),
#             )

#             if j == i:
#                 s_high_value = make_or(nodes[i - 1][j - 1].high, nodes[i][j - 1].high)
#                 statements.append(make_assignment(s_high_name, s_high_value))
#                 s_low_value = make_and(nodes[i - 1][j - 1].high, nodes[i][j - 1].high)
#                 statements.append(make_assignment(s_low_name, s_low_value))
#             else:
#                 s_high_value = make_or(nodes[i - 1][j].low, nodes[i][j - 1].high)
#                 statements.append(make_assignment(s_high_name, s_high_value))
#                 s_low_value = make_and(nodes[i - 1][j].low, nodes[i][j - 1].high)
#                 statements.append(make_assignment(s_low_name, s_low_value))

#     # Step 4: Determine output nodes (AST nodes representing the sorted results)
#     # outputs = [largest, ..., smallest] (confirmed descending)
#     outputs_nodes = [nodes[n_val - 1][n_val - 1].high] + [
#         nodes[n_val - 1][i].low for i in range(n_val - 1, 0, -1)
#     ]

#     # --- Step 5: FIXED Assignment ---
#     # k_target = k_val - 1  # Target number of ones is k

#     # Handle edge case k=0 or k=1 where boundary check isn't needed or possible
#     # if n_val <= 1:
#     #     if n_val == 1 and k_target == 0:  # Check for exactly 0 ones for n=1
#     #         idx0 = 0
#     #         var0_name = "sorted_bit_0"
#     #         statements.append(
#     #             make_assignment(
#     #                 var0_name, ast.UnaryOp(op=ast.Invert(), operand=outputs_nodes[idx0])
#     #             )
#     #         )
#     #         # We only need one bit for this check; make sorted_bit_1 always 0
#     #         var1_name = "sorted_bit_1"
#     #         statements.append(
#     #             make_assignment(var1_name, ast.Constant(value=0))
#     #         )  # Assign constant 0 (adjust if using BitVec(1,0))
#     #     # If k_target is 1 for n=1, or n=0, the current check likely doesn't apply well.
#     #     # For simplicity, let's assume n > 1 for the boundary check.
#     #     # If you need k=0 or k=1 checks, the logic in parameterized_clique_counter might need adjustment too.
#     #     print(f"Warning: Boundary check might be ill-defined for k={n_val}")

#     # else:

#     # Index of the last expected '1' (k_target-th element, 0-indexed)
#     boundary_index_1 = k_val - 1

#     # Ensure indices are valid for the outputs_nodes list
#     if 0 <= boundary_index_1 < len(outputs_nodes):
#         # Assign output[k-1] to sorted_bit_0
#         var0_name = "sorted_bit_0"
#         statements.append(make_assignment(var0_name, outputs_nodes[boundary_index_1]))

#     else:
#         # This shouldn't happen for n > 1 if k = floor(n/2)
#         raise IndexError(
#             f"Calculated boundary indicex {boundary_index_1}are out of range for sorted list of length {len(outputs_nodes)} (k_val={n_val})"
#         )

#     # The function returns the list of statements.
#     # The calling function `parameterized_clique_counter` uses the variables
#     # `sorted_bit_0` and `sorted_bit_1` created here.
#     return statements

import ast
import copy
from typing import Any, Dict, List, Optional, Union

# Assuming BitVec is defined elsewhere, if needed for type hints
# from tweedledum import BitVec


# Helper to resolve classical parameters from AST nodes
def _resolve_classical_value(
    param_node: Union[ast.Name, ast.Constant],
    param_name: str,
    classical_inputs: Optional[Dict[str, Any]],
) -> int:
    """
    Resolves an AST node (Constant or Name) to its integer value.

    Args:
        param_node: The AST node representing the parameter.
        param_name: The expected name of the parameter (for error messages).
        classical_inputs: Dictionary holding values for named parameters.

    Returns:
        The integer value of the parameter.

    Raises:
        ValueError: If the parameter cannot be resolved or is not an integer.
        TypeError: If classical_inputs is needed but not provided.
    """
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


# --- AST Node Creation Helpers ---


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
    # Ensure vector name is Load context
    vec_load = copy.deepcopy(vec_node)
    vec_load.ctx = ast.Load()
    return ast.Subscript(
        value=vec_load,
        slice=_make_constant(value=index_val),
        ctx=ast.Load(),
    )


def _make_assignment(target_var_name: str, value_node: ast.expr) -> ast.Assign:
    """Creates an AST Assignment node target = value."""
    # Ensure value is Load context
    value_load = copy.deepcopy(value_node)
    if hasattr(value_load, "ctx"):  # Constants don't have ctx
        value_load.ctx = ast.Load()

    assign_node = ast.Assign(
        targets=[_make_store_name(target_var_name)], value=value_load
    )
    # Ensure generated AST nodes have location info for unparsing
    ast.fix_missing_locations(assign_node)
    return assign_node


def _make_binary_op(
    left_node: ast.expr, right_node: ast.expr, op: ast.operator
) -> ast.BinOp:
    """Creates an AST BinOp node left op right."""
    # Ensure operands are Load context
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


# --- Main Function ---


class SortPairNode:
    """Node for sorting network storing AST nodes for high/low values."""

    def __init__(self, high: Optional[ast.expr], low: Optional[ast.expr]):
        self.high = high
        self.low = low


def generate_sorting_network(
    input_vec_node: ast.Name,
    n_node: Union[ast.Name, ast.Constant],
    k_node: Union[ast.Name, ast.Constant],
    classical_inputs: Optional[Dict[str, Any]] = None,
) -> List[ast.Assign]:
    """
    Generates AST assignment statements for a sorting network.

    The network sorts input bits (from input_vec_node) in descending order (1s first).
    It assigns the bit corresponding to the threshold for having at least 'k' ones
    to the variable "sorted_bit_0".

    Args:
        input_vec_node: AST Name node for the input BitVec variable (e.g., vertices).
        n_node: AST Constant or Name node representing the total number of vertices (n).
        k_node: AST Constant or Name node representing the minimum clique size (k).
        classical_inputs: Dictionary providing values for n_node/k_node if they are Names.

    Returns:
        A list of ast.Assign nodes representing the sorting network logic.

    Raises:
        ValueError: If input validation fails or parameters are inconsistent.
        TypeError: If AST node types are incorrect or classical_inputs is missing when needed.
    """
    # 1. Resolve and Validate Parameters
    if not isinstance(input_vec_node, ast.Name):
        raise TypeError("Input 'input_vec_node' must be an ast.Name node.")
    input_vec_name = input_vec_node.id

    num_vertices = _resolve_classical_value(n_node, "n", classical_inputs)
    min_clique_size = _resolve_classical_value(k_node, "k", classical_inputs)

    if not isinstance(num_vertices, int) or num_vertices <= 0:
        raise ValueError(
            f"Number of vertices 'n' must be a positive integer, got {num_vertices}."
        )
    if not isinstance(min_clique_size, int) or min_clique_size < 1:  # Allow k=1 check
        raise ValueError(
            f"Minimum clique size 'k' must be >= 1, got {min_clique_size}."
        )
    if min_clique_size > num_vertices:
        raise ValueError(
            f"Cannot find clique size {min_clique_size} with only {num_vertices} vertices."
        )

    # 2. Initialize
    statements: List[ast.Assign] = []
    # Use a more descriptive name internally if desired
    vec_ast_node = _make_load_name(input_vec_name)
    input_bit_nodes = [_make_subscript(vec_ast_node, i) for i in range(num_vertices)]

    # Comparator network stages (using SortPairNode to hold AST expressions)
    # Using num_vertices x num_vertices grid, though fewer stages might suffice
    stages: List[List[SortPairNode]] = [
        [SortPairNode(None, None) for _ in range(num_vertices)]
        for _ in range(num_vertices)
    ]

    # Load initial inputs into the first stage (conceptually column 0)
    for i in range(num_vertices):
        stages[i][0] = SortPairNode(input_bit_nodes[i], None)

    # 3. Generate Sorting Network Logic (Comparator Stages)
    # This implements a standard comparator-based sorting network structure.
    # The specific connections depend on the chosen algorithm (e.g., Batcher's).
    # The high/low logic implements a compare-and-swap (max goes to high, min to low).
    for i in range(1, num_vertices):  # Stage index (or conceptual row)
        for j in range(1, i + 1):  # Comparator index within stage (conceptual col)
            # Names for intermediate variables holding results of this comparator
            s_high_name = f"s_{i}_{j}_high"
            s_low_name = f"s_{i}_{j}_low"

            # Store AST nodes representing these intermediate variables
            current_high_node = _make_load_name(s_high_name)
            current_low_node = _make_load_name(s_low_name)
            stages[i][j] = SortPairNode(current_high_node, current_low_node)

            # Determine inputs to this comparator based on network structure
            if (
                j == i
            ):  # Comparator connects diagonally adjacent elements from previous stage
                prev_high = stages[i - 1][j - 1].high
                prev_diag_high = stages[i][
                    j - 1
                ].high  # From same stage, previous comparator
                # Calculate High Output: H = In1 OR In2
                s_high_value_ast = _make_or(prev_high, prev_diag_high)
                statements.append(_make_assignment(s_high_name, s_high_value_ast))
                # Calculate Low Output: L = In1 AND In2
                s_low_value_ast = _make_and(prev_high, prev_diag_high)
                statements.append(_make_assignment(s_low_name, s_low_value_ast))
            else:  # Comparator connects vertically adjacent elements from previous stage
                prev_low = stages[i - 1][
                    j
                ].low  # From previous stage, same comparator index
                prev_diag_high = stages[i][
                    j - 1
                ].high  # From same stage, previous comparator
                # Calculate High Output: H = In1 OR In2
                s_high_value_ast = _make_or(prev_low, prev_diag_high)
                statements.append(_make_assignment(s_high_name, s_high_value_ast))
                # Calculate Low Output: L = In1 AND In2
                s_low_value_ast = _make_and(prev_low, prev_diag_high)
                statements.append(_make_assignment(s_low_name, s_low_value_ast))

    # 4. Identify Final Sorted Output Nodes
    # Based on structure, outputs are gathered from the last stage (descending order)
    # outputs_nodes[0] = largest, outputs_nodes[n-1] = smallest
    final_stage = num_vertices - 1
    outputs_nodes = [stages[final_stage][final_stage].high] + [
        stages[final_stage][i].low for i in range(final_stage, 0, -1)
    ]

    # 5. Assign the Critical Boundary Bit
    # To check for *at least* k ones (popcount >= k), we check if the k-th bit
    # (index k-1) in the descending sorted list is 1.
    boundary_index = min_clique_size - 1

    if not (0 <= boundary_index < len(outputs_nodes)):
        # This should only happen if k > n or k < 1, already checked
        raise IndexError(
            f"Calculated boundary index {boundary_index} is out of range for "
            f"sorted list of length {len(outputs_nodes)} (n={num_vertices}, k={min_clique_size})"
        )

    # Assign the AST node for the k-th largest bit to the specific variable name "sorted_bit_0"
    target_output_node = outputs_nodes[boundary_index]
    result_var_name = "sorted_bit_0"  # Hardcoded name expected by the calling function
    statements.append(_make_assignment(result_var_name, target_output_node))

    return statements
