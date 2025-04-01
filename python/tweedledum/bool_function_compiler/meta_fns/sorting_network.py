import ast


class SortPairNode:
    """Node for sorting network that tracks high and low values."""

    def __init__(self, high, low):
        self.high = high
        self.low = low


def generate_sorting_network(input_vec, output_name, k_node):
    """
    Generate sorting network AST statements with intermediate variables.

    This approach:
    1. Creates a BitVec initialized with all 1s
    2. Uses the standard sorting network algorithm
    3. Updates output bits using AND operations
    4. Creates separate variables for each bit for safe use in return statements

    Args:
        input_vec (ast.Name): AST node for input BitVec variable
        output_name (ast.Constant): AST node for the output variable name
        k_node (Union[ast.Name, ast.Constant]): AST node for the size - can be variable or constant

    Returns:
        list: List of AST statement nodes
    """
    # Validate input types
    if not isinstance(input_vec, ast.Name):
        raise ValueError("Input must be a variable name")

    if not isinstance(output_name, ast.Constant):
        raise ValueError("Output name must be a constant string")

    output_var_name = output_name.value

    # Extract size value if it's a constant, otherwise use the variable directly
    if isinstance(k_node, ast.Constant):
        k = k_node.value
    elif isinstance(k_node, ast.Name):
        # We'll need to get the value at runtime through the variable
        k_name = k_node.id
        # Look for this variable in the classical_inputs from the transformer
        if (
            hasattr(generate_sorting_network, "classical_inputs")
            and k_name in generate_sorting_network.classical_inputs
        ):
            k = generate_sorting_network.classical_inputs[k_name]
        else:
            raise ValueError(
                f"Unable to determine size from variable {k_name}. Make sure it's a classical input."
            )
    else:
        raise ValueError("k must be either a constant or a variable name")

    input_name = input_vec.id

    # Create helper functions for AST creation
    def make_subscript(name, index):
        return ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()),
            slice=ast.Constant(value=index),
            ctx=ast.Load(),
        )

    def make_assignment(target, value):
        if isinstance(target, str):
            target_node = ast.Name(id=target, ctx=ast.Store())
        else:
            target_node = target
            target_node.ctx = ast.Store()
        return ast.Assign(targets=[target_node], value=value)

    def make_or(left, right):
        if isinstance(left, str):
            left = ast.Name(id=left, ctx=ast.Load())
        if isinstance(right, str):
            right = ast.Name(id=right, ctx=ast.Load())
        return ast.BinOp(left=left, op=ast.BitOr(), right=right)

    def make_and(left, right):
        if isinstance(left, str):
            left = ast.Name(id=left, ctx=ast.Load())
        if isinstance(right, str):
            right = ast.Name(id=right, ctx=ast.Load())
        return ast.BinOp(left=left, op=ast.BitAnd(), right=right)

    # Initialize statements list
    statements = []

    # Step 2: Create input variables references
    input_vars = [make_subscript(input_name, i) for i in range(k)]

    # Step 3: Initialize sorting network exactly as in the original algorithm
    nodes = [[SortPairNode(None, None) for _ in range(k)] for _ in range(k)]

    # Initialize with input variables
    for i in range(k):
        nodes[i][0] = SortPairNode(input_vars[i], None)

    # Generate sorting network logic
    for i in range(1, k):
        for j in range(1, i + 1):
            s_high_name = f"s_{i}_{j}_high"
            s_low_name = f"s_{i}_{j}_low"

            nodes[i][j] = SortPairNode(
                ast.Name(id=s_high_name, ctx=ast.Load()),
                ast.Name(id=s_low_name, ctx=ast.Load()),
            )

            if j == i:
                # s_high = prev_high OR prev_diag_high
                s_high_value = make_or(nodes[i - 1][j - 1].high, nodes[i][j - 1].high)
                statements.append(make_assignment(s_high_name, s_high_value))

                # s_low = prev_high AND prev_diag_high
                s_low_value = make_and(nodes[i - 1][j - 1].high, nodes[i][j - 1].high)
                statements.append(make_assignment(s_low_name, s_low_value))
            else:
                # s_high = prev_low OR prev_diag_high
                s_high_value = make_or(nodes[i - 1][j].low, nodes[i][j - 1].high)
                statements.append(make_assignment(s_high_name, s_high_value))

                # s_low = prev_low AND prev_diag_high
                s_low_value = make_and(nodes[i - 1][j].low, nodes[i][j - 1].high)
                statements.append(make_assignment(s_low_name, s_low_value))

    # Step 4: Determine output nodes in the correct order
    outputs = [nodes[k - 1][k - 1].high] + [
        nodes[k - 1][i].low for i in range(k - 1, 0, -1)
    ]

    # Step 5: Create separate variables for each bit to use in return statements
    # Calculate middle indices dynamically based on k
    middle_indices = [k // 2 - 1, k // 2] if k > 1 else [0]
    for idx, bit in enumerate(middle_indices):
        if 0 <= idx < len(outputs):  # Ensure index is valid
            bit_var_name = f"sorted_bit_{idx}"
            statements.append(make_assignment(bit_var_name, outputs[bit]))

    return statements
