# -------------------------------------------------------------------------------
# Part of Tweedledum Project.  This file is distributed under the MIT License.
# See accompanying file /LICENSE for details.
# -------------------------------------------------------------------------------
import _ast
import ast

import astunparse

from tweedledum.classical import LogicNetwork

from .bitvec import BitVec
from .meta_fns import _global_generators


class ParseError(Exception):
    pass


class FunctionParser(ast.NodeVisitor):
    bit_ops = {
        _ast.BitAnd: "create_and",
        _ast.BitOr: "create_or",
        _ast.BitXor: "create_xor",
    }

    bool_ops = {
        _ast.And: "create_and",
        _ast.Or: "create_or",
    }

    # This feels quite hack-y
    types = {"BitVec": type(BitVec(1))}

    def __init__(self, source):
        self._symbol_table = dict()
        self._parameters_signature = list()
        self._return_signature = list()
        self._logic_network = LogicNetwork()

        node = ast.parse(source)
        self.visit(node)
        super().__init__()

    def visit_args(self, node):
        cache = list()
        for arg in node.args:
            cache.append(arg.arg)
            if arg.annotation is None:
                continue
            if not isinstance(arg.annotation, _ast.Call):
                raise ParseError("BitVec type with size is needed")
            for var in cache:
                pis = list()
                arg_type, arg_size = self._visit_annotation_Call(arg.annotation)
                for i in range(arg_size):
                    pis.append(self._logic_network.create_pi("{}_{}".format(var, i)))
                self._symbol_table[var] = ((arg_type, arg_size), pis)
                self._parameters_signature.append((arg_type, arg_size))
            cache.clear()
        if len(cache) != 0:
            raise ParseError("Argument type is needed for %s" % cache)

    # def visit_Assign(self, node):
    #     """When assign, the scope needs to be updated with the right type"""
    #     value_type, value_signals = self.visit(node.value)
    #     for target in node.targets:
    #         self._symbol_table[target.id] = (value_type, value_signals)
    #     return (value_type, value_signals)

    def visit_Assign(self, node):
        """When assign, the scope needs to be updated with the right type.
        Handles assignment to both simple variables and single BitVec elements.
        """
        # Visit the right-hand side value first to get its signals and type
        value_type, value_signals = self.visit(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                # --- Standard variable assignment ---
                # Update symbol table to map variable name to the new signals
                self._symbol_table[target.id] = (value_type, value_signals)
                # --- End Standard variable assignment ---

            elif isinstance(target, ast.Subscript):
                # --- BitVec single element assignment ---
                if not isinstance(target.value, ast.Name):
                    # Ensure the base of the subscript is a variable name
                    raise ParseError(
                        f"Cannot assign to subscript of non-variable: {astunparse.unparse(target.value)}"
                    )

                base_var_name = target.value.id
                if base_var_name not in self._symbol_table:
                    raise ParseError(
                        f"Assigning to subscript of unknown variable: {base_var_name}"
                    )

                # Get current signals and type info for the base BitVec
                base_type_info, current_signals = self._symbol_table[base_var_name]
                base_type, base_length = base_type_info  # Unpack type and length

                # Determine the target index
                target_index_or_slice = self.visit(target.slice)

                # Create a mutable copy of the current signals
                new_signal_list = list(current_signals)

                if isinstance(target_index_or_slice, int):
                    # --- Assignment to a single index ---
                    target_index = target_index_or_slice
                    if not (0 <= target_index < base_length):
                        raise ParseError(
                            f"Index {target_index} out of bounds for {base_var_name} of length {base_length}"
                        )
                    if len(value_signals) != 1:
                        raise ParseError(
                            f"Assigning multiple signals ({len(value_signals)}) to single index {target_index} of {base_var_name}"
                        )

                    # Replace the signal at the target index (Python list index matches BitVec LSB index)
                    new_signal_list[target_index] = value_signals[0]
                    # --- End Assignment to a single index ---

                else:
                    # Only integer indices are supported now
                    raise ParseError(
                        f"Unsupported type {type(target_index_or_slice)} for BitVec assignment index. Only integer indices are supported."
                    )

                # Update symbol table for the base variable name to point to the *new* list of signals
                self._symbol_table[base_var_name] = (base_type_info, new_signal_list)
                # --- End BitVec single element assignment ---

            else:
                # Handle other potential target types if necessary, or raise error
                raise ParseError(f"Unsupported assignment target type: {type(target)}")

        # Return the type and signals of the value that was assigned
        return (value_type, value_signals)

    def visit_BinOp(self, node):
        """Handles ``&``, ``^``, and ``|``."""
        op = FunctionParser.bit_ops.get(type(node.op))
        if not op:
            raise ParseError("Unknown binop.op %s" % op)
        left_type, left_signals = self.visit(node.left)
        right_type, right_signals = self.visit(node.right)
        if len(left_signals) != len(right_signals):
            raise ParseError("Different length binop.op %s" % op)
        result = list()
        for l, r in zip(left_signals, right_signals):
            result.append(getattr(self._logic_network, op)(l, r))
        return right_type, result

    def visit_BoolOp(self, node):
        _, result_signal = self.visit(node.values[0])
        op = FunctionParser.bool_ops.get(type(node.op))
        for value in node.values[1:]:
            _, value_signal = self.visit(value)
            result_signal[0] = getattr(self._logic_network, op)(
                result_signal[0], value_signal[0]
            )
        return (FunctionParser.types["BitVec"], 1), result_signal

    def visit_Call(self, node):
        if node.func.id in _global_generators:
            return None

        type_ = FunctionParser.types[node.func.id]
        if len(node.args) == 1:
            value = ast.literal_eval(node.args[0])
            if isinstance(value, str):
                return (type_, len(value)), self._const_BitVec(len(value), value)
            elif isinstance(value, int):
                return (type_, value), self._const_BitVec(value)
        elif len(node.args) == 2:
            length = ast.literal_eval(node.args[0])
            if not isinstance(length, int):
                raise ParseError("BitVec requires length to be an integer")
            value = ast.literal_eval(node.args[1])
            if isinstance(value, int):
                value = "{:0{}b}".format(value, length)
            return (type_, length), self._const_BitVec(length, value)
        else:
            raise ParseError("Invalid number of arguments")

    def visit_Compare(self, node):
        left_type, left_signals = self.visit(node.left)
        partial_results = list()
        for op, right in zip(node.ops, node.comparators):
            right_type, right_signals = self.visit(right)
            if len(left_signals) != len(right_signals):
                raise ParseError("Different length")
            if isinstance(op, ast.Eq):
                new_signals = [
                    self._logic_network.create_xnor(i, j)
                    for i, j in zip(left_signals, right_signals)
                ]
                partial_results.append(self._logic_network.create_nary_and(new_signals))
            elif isinstance(op, ast.NotEq):
                new_signals = [
                    self._logic_network.create_xor(i, j)
                    for i, j in zip(left_signals, right_signals)
                ]
                partial_results.append(self._logic_network.create_nary_or(new_signals))
            else:
                raise ParseError("Unsupported comparator")

        result = self._logic_network.create_nary_and(partial_results)
        return (FunctionParser.types["BitVec"], 1), [result]

    def visit_FunctionDef(self, node):
        if node.returns is None:
            raise ParseError("Return type is needed")
        if isinstance(node.returns, _ast.Call):
            return_type = node.returns.func.id
            size = ast.literal_eval(node.returns.args[0])
            if not isinstance(size, int):
                ParseError("Size must be an integer")
            self._return_signature = [(FunctionParser.types[return_type], size)]
            self._symbol_table["__dee_ret_0"] = (return_type, None)
        elif isinstance(node.returns, ast.Tuple):
            for i, elt in enumerate(node.returns.elts):
                return_type = elt.func.id
                size = ast.literal_eval(elt.args[0])
                if not isinstance(size, int):
                    ParseError("Size must be an integer")
                self._return_signature.append((FunctionParser.types[return_type], size))
                self._symbol_table[f"__dee_ret_{i}"] = (return_type, None)
        else:
            raise ParseError("Return type is needed")

        self.visit_args(node.args)
        return super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self._symbol_table:
            raise ParseError("out of scope: %s" % node.id)
        return self._symbol_table[node.id][0], self._symbol_table[node.id][1]

    def visit_Return(self, node):
        """The return type should match the return type hint."""
        if isinstance(node.value, ast.Tuple):
            if len(self._return_signature) != len(node.value.elts):
                raise ParseError(
                    f"The function was expected to return "
                    f"{len(self._return_signature)} values, "
                    f"but it returned {len(node.value.elts)}."
                )
            for i, elt in enumerate(node.value.elts):
                elt_type, elt_signals = self.visit(elt)
                if elt_type != self._return_signature[i]:
                    raise ParseError(
                        f"The {i}-th return value was expected to "
                        f"be of type {self._return_signature[i]}, "
                        f"but got {elt_type}."
                    )

                for s in elt_signals:
                    self._logic_network.create_po(s)
        else:
            if len(self._return_signature) > 1:
                raise ParseError(
                    f"The function was expected to return "
                    f"{len(self._return_signature)} values, "
                    f"but it returned just 1."
                )
            return_type, signals = self.visit(node.value)
            if return_type != self._return_signature[0]:
                raise ParseError(
                    f"The return value was expected to "
                    f"be of type {self._return_signature[0]}, "
                    f"but got {return_type}."
                )
            for s in signals:
                self._logic_network.create_po(s)

    def visit_Slice(self, node):
        lower = ast.literal_eval(node.lower)
        upper = ast.literal_eval(node.lower)
        return slice(lower, upper)

    # Python < 3.9
    def visit_Index(self, node):
        if not isinstance(node.value, ast.Num):
            raise ParseError("Subscript index must be a number")
        return ast.literal_eval(node.value)

    def visit_Constant(self, node):
        return ast.literal_eval(node)

    def visit_Subscript(self, node):
        try:
            v_type, v_signals = self.visit(node.value)
            slice_ = self.visit(node.slice)
            if isinstance(slice_, int):
                # Return the correct type tuple format, not a BitVec object
                return (FunctionParser.types["BitVec"], 1), [v_signals[slice_]]
            if isinstance(slice_, slice):
                return v_type, v_signals[slice_]
            raise ParseError("Subscript must be an integer or a slice")
        except Exception as _:
            print(f"ERROR in visit_Subscript for node: {ast.dump(node)}")
            raise

    # def visit_Subscript(self, node):
    #     v_type, v_signals = self.visit(node.value)
    #     slice_ = self.visit(node.slice)
    #     # FIXME: The v_types here need to be changes to proper bit width
    #     if isinstance(slice_, int):
    #         return (FunctionParser.types["BitVec"], 1), [v_signals[slice_]]
    #     if isinstance(slice_, slice):
    #         return v_type, v_signals[slice_]
    #     raise ParseError("Subscript must be an integer or a slice")

    def visit_UnaryOp(self, node):
        result_type, result_signal = self.visit(node.operand)
        if isinstance(node.op, _ast.Not):
            if len(result_signal) != 1:
                raise ParseError("Boolean NOT doesn't work on multibit values")
            return result_type, [self._logic_network.create_not(result_signal[0])]
        elif isinstance(node.op, ast.Invert):
            result = list()
            for s in result_signal:
                result.append(self._logic_network.create_not(s))
            return result_type, result
        raise ParseError("Unsupported unary operator")

    def _visit_annotation_Call(self, node):
        # this only works for BitVec arguments
        type_ = FunctionParser.types[node.func.id]
        # if len(node.args) == 1:
        size = ast.literal_eval(node.args[0])
        if not isinstance(size, int):
            ParseError("Size must be an integer")
        return type_, size
        # raise ParseError("Invalid number of arguments")

    def _const_BitVec(self, length, value=None):
        if value == None:
            return [self._logic_network.get_constant(0) for i in range(length)]
        if not isinstance(value, str):
            raise ParseError(
                "When creating constant BitVec, value must be a string or None"
            )
        if len(value) > length:
            ParseError(
                f"BitVec value requires a bit vector of "
                f"length {len(value)}, but declared BitVec has "
                f"length {length}"
            )
        return [self._logic_network.get_constant(i == "1") for i in value[::-1]]
