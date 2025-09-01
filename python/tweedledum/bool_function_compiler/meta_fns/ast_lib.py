import ast
import copy
from typing import Any, Dict, Optional, Union

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
