import ast


def generate_binary_popcount(input_vec, output_name, k_node):
    """
    Generate AST statements for efficiently counting 1-bits using a binary tree approach.

    This function creates a binary tree of addition operations to count the
    number of set bits in the input vector, then checks if that count equals k/2.

    Args:
        input_vec (ast.Name): AST node for input BitVec variable
        output_name (ast.Constant): AST node for the output variable name as a string
        k_node (ast.Constant): AST node for the size constant (must be power of 2)

    Returns:
        list: List of AST statement nodes
    """
    # Validate input types
    if not isinstance(input_vec, ast.Name):
        raise ValueError("Input must be a variable name (ast.Name node)")

    if not isinstance(output_name, ast.Constant) or not isinstance(
        output_name.value, str
    ):
        raise ValueError("Output name must be a constant string")

    # Extract size value from AST node
    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, int):
        k = k_node.value
    else:
        raise ValueError("k must be a constant integer value")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    input_name = input_vec.id
    output_var_name = output_name.value

    # Create helper functions for AST node creation
    def make_subscript(name, index):
        """Create an AST node for a subscript expression: name[index]"""
        if isinstance(name, str):
            name_node = ast.Name(id=name, ctx=ast.Load())
        else:
            name_node = name

        return ast.Subscript(
            value=name_node,
            slice=ast.Constant(value=index)
            if hasattr(ast, "Constant")
            else ast.Index(value=ast.Num(n=index)),
            ctx=ast.Load(),
        )

    def make_assignment(target, value):
        """Create an AST node for an assignment: target = value"""
        if isinstance(target, str):
            target_node = ast.Name(id=target, ctx=ast.Store())
        else:
            target_node = target
            target_node.ctx = ast.Store()

        return ast.Assign(targets=[target_node], value=value)

    def make_bitvec(length, value=0):
        """Create an AST node for BitVec creation: BitVec(length, value)"""
        return ast.Call(
            func=ast.Name(id="BitVec", ctx=ast.Load()),
            args=[
                ast.Constant(value=length)
                if hasattr(ast, "Constant")
                else ast.Num(n=length),
                ast.Constant(value=value)
                if hasattr(ast, "Constant")
                else ast.Num(n=value),
            ],
            keywords=[],
        )

    def make_add(left, right):
        """Create an AST node for addition: left + right"""
        if isinstance(left, str):
            left = ast.Name(id=left, ctx=ast.Load())
        if isinstance(right, str):
            right = ast.Name(id=right, ctx=ast.Load())

        return ast.BinOp(left=left, op=ast.Add(), right=right)

    def make_eq(left, right):
        """Create an AST node for equality comparison: left == right"""
        if isinstance(left, str):
            left = ast.Name(id=left, ctx=ast.Load())
        if isinstance(right, str):
            right = ast.Name(id=right, ctx=ast.Load())

        return ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])

    def make_and(left, right):
        """Create an AST node for bitwise AND: left & right"""
        if isinstance(left, str):
            left = ast.Name(id=left, ctx=ast.Load())
        if isinstance(right, str):
            right = ast.Name(id=right, ctx=ast.Load())

        return ast.BinOp(left=left, op=ast.BitAnd(), right=right)

    def make_not(expr):
        """Create an AST node for bitwise NOT: ~expr"""
        if isinstance(expr, str):
            expr = ast.Name(id=expr, ctx=ast.Load())

        return ast.UnaryOp(op=ast.Invert(), operand=expr)

    # Initialize statements list
    statements = []

    # Determine the number of bits needed to represent the count
    # Maximum count is k, so we need log₂(k+1) bits rounded up
    import math

    count_bits = math.ceil(math.log2(k + 1))

    # Initialize count variable with zeros
    count_var = f"{output_var_name}_count"
    statements.append(make_assignment(count_var, make_bitvec(count_bits, 0)))

    # First layer: copy inputs to working variables
    bits = [f"bit_{i}" for i in range(k)]
    for i in range(k):
        statements.append(make_assignment(bits[i], make_subscript(input_name, i)))

    # Binary tree addition
    layer = 0
    while len(bits) > 1:
        new_bits = []
        for i in range(0, len(bits), 2):
            if i + 1 < len(bits):
                # Add pair of bits and store result
                sum_var = f"sum_{layer}_{i // 2}"
                statements.append(
                    make_assignment(sum_var, make_add(bits[i], bits[i + 1]))
                )
                new_bits.append(sum_var)
            else:
                # Odd number of bits, just pass through
                pass_var = f"pass_{layer}_{i // 2}"
                statements.append(
                    make_assignment(pass_var, ast.Name(id=bits[i], ctx=ast.Load()))
                )
                new_bits.append(pass_var)
        bits = new_bits
        layer += 1

    # Store the final sum in count variable
    statements.append(make_assignment(count_var, ast.Name(id=bits[0], ctx=ast.Load())))

    # Check if count equals k/2
    target = k // 2
    statements.append(
        make_assignment(
            output_var_name,
            make_eq(
                count_var,
                ast.Constant(value=target)
                if hasattr(ast, "Constant")
                else ast.Num(n=target),
            ),
        )
    )

    return statements
