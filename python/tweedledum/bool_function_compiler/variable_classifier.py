import ast
import logging
from typing import Any, Dict

# logger named 'variable_classifier'
logger = logging.getLogger(__name__)


class VariableClassifier(ast.NodeVisitor):
    """
    Classifies variables as quantum or classical based on AST analysis.
    """

    def __init__(
        self,
        quantum_params: Dict[str, Any],
        classical_inputs: Dict[str, Any],
        debug=False,
    ):
        if debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(level=logging.INFO)

        self.quantum_params = set(quantum_params.keys())
        self.classical_inputs = set(classical_inputs.keys())

        # Sets to track variable classifications
        self.quantum_vars = set(quantum_params.keys())
        self.classical_vars = set(classical_inputs.keys())
        self.derived_quantum_vars = set()

        # Track expressions that can't be statically evaluated
        self.dynamic_expressions = set()

        logger.debug(f"Initialized with quantum params: {self.quantum_params}")
        logger.debug(f"Initialized with classical inputs: {self.classical_inputs}")

    def visit_Assign(self, node):
        """Classify variables from assignment statements."""
        # Process right side first
        self.visit(node.value)

        # Determine if right side is a quantum expression
        is_quantum_expr = self._is_quantum_expr(node.value)

        # Log the assignment and classification
        if hasattr(node, "lineno"):
            line_info = f" (line {node.lineno})"
        else:
            line_info = ""

        try:
            value_str = ast.unparse(node.value)
            logger.debug(
                f"Assignment{line_info}: {value_str} -> {'quantum' if is_quantum_expr else 'classical'}"
            )
        except:
            logger.debug(
                f"Assignment{line_info}: <complex expr> -> {'quantum' if is_quantum_expr else 'classical'}"
            )

        # Update variable classifications for target variables
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                old_classification = (
                    "quantum"
                    if (
                        var_name in self.quantum_vars
                        or var_name in self.derived_quantum_vars
                    )
                    else "classical"
                )

                if is_quantum_expr:
                    self.derived_quantum_vars.add(var_name)
                    if var_name in self.classical_vars:
                        self.classical_vars.remove(var_name)
                        logger.debug(
                            f"Reclassifying {var_name} from classical to quantum"
                        )
                else:
                    self.classical_vars.add(var_name)

                new_classification = (
                    "quantum"
                    if (
                        var_name in self.quantum_vars
                        or var_name in self.derived_quantum_vars
                    )
                    else "classical"
                )
                if old_classification != new_classification:
                    logger.info(
                        f"Variable {var_name} changed from {old_classification} to {new_classification}"
                    )
                else:
                    logger.debug(
                        f"Variable {var_name} classified as {new_classification}"
                    )

            elif isinstance(target, ast.Subscript) and isinstance(
                target.value, ast.Name
            ):
                # Handle BitVec subscript assignment
                array_name = target.value.id
                if (
                    array_name in self.quantum_vars
                    or array_name in self.derived_quantum_vars
                ):
                    try:
                        index_str = ast.unparse(target.slice)
                        logger.debug(
                            f"Quantum subscript assignment: {array_name}[{index_str}]"
                        )
                    except:
                        logger.debug(
                            f"Quantum subscript assignment: {array_name}[<complex index>]"
                        )

        self.generic_visit(node)

    def visit_Call(self, node):
        """Handle function calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "BitVec":
            # BitVec constructor always creates quantum variables
            self.dynamic_expressions.add(node)
            try:
                args_str = ", ".join(ast.unparse(arg) for arg in node.args)
                logger.debug(f"BitVec constructor: BitVec({args_str}) -> quantum")
            except:
                logger.debug(f"BitVec constructor: BitVec(...) -> quantum")

        self.generic_visit(node)

    def visit_BinOp(self, node):
        """Process binary operations."""
        self.visit(node.left)
        self.visit(node.right)

        # If either operand is quantum, result is quantum
        left_quantum = self._is_quantum_expr(node.left)
        right_quantum = self._is_quantum_expr(node.right)

        if left_quantum or right_quantum:
            self.dynamic_expressions.add(node)
            try:
                op_str = {
                    ast.Add: "+",
                    ast.Sub: "-",
                    ast.Mult: "*",
                    ast.Div: "/",
                    ast.BitOr: "|",
                    ast.BitAnd: "&",
                    ast.BitXor: "^",
                }.get(type(node.op), "?")

                left_str = ast.unparse(node.left)
                right_str = ast.unparse(node.right)

                logger.debug(
                    f"Binary operation: {left_str}({('quantum' if left_quantum else 'classical')}) "
                    f"{op_str} {right_str}({('quantum' if right_quantum else 'classical')}) -> quantum"
                )
            except:
                logger.debug(f"Complex binary operation -> quantum")

    def visit_For(self, node):
        """Handle for loops."""
        # Visit the iterable expression
        self.visit(node.iter)

        # Handle loop variable
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id

            # Check if this is a range-based loop (classical)
            is_classical_loop = (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            )

            if is_classical_loop:
                self.classical_vars.add(loop_var)
                try:
                    range_args = ", ".join(ast.unparse(arg) for arg in node.iter.args)
                    logger.debug(
                        f"Loop variable {loop_var} in range({range_args}) -> classical"
                    )
                except:
                    logger.debug(f"Loop variable {loop_var} in range(...) -> classical")
            else:
                # If iterating over quantum values, loop var is quantum
                self.derived_quantum_vars.add(loop_var)
                try:
                    iter_str = ast.unparse(node.iter)
                    logger.debug(f"Loop variable {loop_var} in {iter_str} -> quantum")
                except:
                    logger.debug(
                        f"Loop variable {loop_var} in <complex iterable> -> quantum"
                    )

        # Visit loop body
        logger.debug(f"Entering loop body with {len(node.body)} statements")
        for stmt in node.body:
            self.visit(stmt)
        logger.debug("Exiting loop body")

    def _is_quantum_expr(self, node):
        """Determine if an expression is quantum or classical."""
        if isinstance(node, ast.Name):
            is_quantum = (
                node.id in self.quantum_params or node.id in self.derived_quantum_vars
            )
            return is_quantum

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # BitVec constructor creates quantum variables
            if node.func.id == "BitVec":
                return True

        elif isinstance(node, ast.BinOp):
            # If either operand is quantum, the result is quantum
            return self._is_quantum_expr(node.left) or self._is_quantum_expr(node.right)

        elif node in self.dynamic_expressions:
            return True

        return False

    def summarize(self):
        """Print a summary of all variable classifications."""
        logger.info("=== Variable Classification Summary ===")
        logger.info(f"Quantum parameters: {sorted(self.quantum_params)}")
        logger.info(f"Derived quantum variables: {sorted(self.derived_quantum_vars)}")
        logger.info(f"Classical variables: {sorted(self.classical_vars)}")
        logger.info("====================================")

    def __del__(self):
        """Print summary when object is destroyed."""
        self.summarize()
