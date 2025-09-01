# bool_function_compiler/cnf_converter.py

import ast
from typing import List, Dict, Tuple, Union, Optional, Set
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
import logging

logger = logging.getLogger("tweedledum.bool_function_compiler.cnf_converter")


class BooleanExpressionExtractor(ast.NodeVisitor):
    """
    Extracts Boolean expressions from the transformed AST and builds
    a symbolic representation suitable for CNF conversion.
    """

    def __init__(self):
        self.expressions = []  # List of (var_name, expression_ast) tuples
        self.input_vars = set()  # Set of input variable names
        self.intermediate_vars = set()  # Set of intermediate variable names
        self.final_output = None  # The final return expression

    def visit_FunctionDef(self, node):
        """Extract input variables from function signature."""
        for arg in node.args.args:
            self.input_vars.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Extract Boolean assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if var_name not in self.input_vars:
                self.intermediate_vars.add(var_name)
            self.expressions.append((var_name, node.value))
        self.generic_visit(node)

    def visit_Return(self, node):
        """Extract the final return expression."""
        self.final_output = node.value


class CNFConverter:
    """
    Converts Boolean expressions to CNF and then to 3-SAT format.
    """

    def __init__(self):
        self.vpool = IDPool()  # Variable pool for mapping names to integers
        self.clauses = []
        self.tseitin_counter = 0  # For generating unique Tseitin variables

    def var_to_int(self, var_name: str, index: Optional[int] = None) -> int:
        """Convert variable name (with optional index) to integer."""
        if index is not None:
            full_name = f"{var_name}[{index}]"
        else:
            full_name = var_name
        return self.vpool.id(full_name)

    def extract_var_from_subscript(self, node: ast.Subscript) -> int:
        """Extract variable integer from subscript node like vars[i]."""
        if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant):
            return self.var_to_int(node.value.id, node.slice.value)
        raise ValueError(f"Unsupported subscript format: {ast.dump(node)}")

    def expr_to_cnf(self, expr: ast.AST, result_var: Optional[int] = None) -> int:
        """
        Convert an expression to CNF using Tseitin transformation.
        Returns the variable representing the expression result.
        """
        if isinstance(expr, ast.Name):
            # Simple variable reference
            return self.var_to_int(expr.id)

        elif isinstance(expr, ast.Subscript):
            # Array subscript like vars[0]
            return self.extract_var_from_subscript(expr)

        elif isinstance(expr, ast.Constant):
            # Boolean constant
            if expr.value == 1 or expr.value == True:
                # Create a new variable that's always true
                true_var = self.vpool.id(f"__true_{self.tseitin_counter}")
                self.tseitin_counter += 1
                self.clauses.append([true_var])  # Unit clause forcing it to be true
                return true_var
            else:
                # Create a new variable that's always false
                false_var = self.vpool.id(f"__false_{self.tseitin_counter}")
                self.tseitin_counter += 1
                self.clauses.append([-false_var])  # Unit clause forcing it to be false
                return false_var

        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Invert):
            # Negation (~x)
            operand_var = self.expr_to_cnf(expr.operand)
            if result_var is None:
                result_var = self.vpool.id(f"__not_{self.tseitin_counter}")
                self.tseitin_counter += 1
            # result_var <-> NOT operand_var
            # CNF: (result_var OR operand_var) AND (NOT result_var OR NOT operand_var)
            self.clauses.append([result_var, operand_var])
            self.clauses.append([-result_var, -operand_var])
            return result_var

        elif isinstance(expr, ast.BinOp):
            left_var = self.expr_to_cnf(expr.left)
            right_var = self.expr_to_cnf(expr.right)

            if result_var is None:
                result_var = self.vpool.id(f"__tseitin_{self.tseitin_counter}")
                self.tseitin_counter += 1

            if isinstance(expr.op, ast.BitOr):
                # result_var <-> (left_var OR right_var)
                # CNF: (NOT result_var OR left_var OR right_var) AND
                #      (result_var OR NOT left_var) AND (result_var OR NOT right_var)
                self.clauses.append([-result_var, left_var, right_var])
                self.clauses.append([result_var, -left_var])
                self.clauses.append([result_var, -right_var])

            elif isinstance(expr.op, ast.BitAnd):
                # result_var <-> (left_var AND right_var)
                # CNF: (NOT result_var OR left_var) AND (NOT result_var OR right_var) AND
                #      (result_var OR NOT left_var OR NOT right_var)
                self.clauses.append([-result_var, left_var])
                self.clauses.append([-result_var, right_var])
                self.clauses.append([result_var, -left_var, -right_var])

            elif isinstance(expr.op, ast.BitXor):
                # result_var <-> (left_var XOR right_var)
                # CNF: (NOT result_var OR left_var OR right_var) AND
                #      (NOT result_var OR NOT left_var OR NOT right_var) AND
                #      (result_var OR left_var OR NOT right_var) AND
                #      (result_var OR NOT left_var OR right_var)
                self.clauses.append([-result_var, left_var, right_var])
                self.clauses.append([-result_var, -left_var, -right_var])
                self.clauses.append([result_var, left_var, -right_var])
                self.clauses.append([result_var, -left_var, right_var])

            return result_var

        elif (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "BitVec"
        ):
            # BitVec constructor - handle the constant value if provided
            if len(expr.args) >= 2:
                # BitVec(size, value)
                value_arg = expr.args[1]
                if isinstance(value_arg, ast.Constant):
                    if isinstance(value_arg.value, int):
                        return self.expr_to_cnf(ast.Constant(value=value_arg.value))
                    elif isinstance(value_arg.value, str):
                        # Binary string like '0001' - just take the LSB for now
                        bit_value = int(value_arg.value[-1])
                        return self.expr_to_cnf(ast.Constant(value=bit_value))
            # Default case
            return self.expr_to_cnf(ast.Constant(value=0))

        else:
            raise ValueError(f"Unsupported expression type: {ast.dump(expr)}")

    def to_3sat(self, clauses: List[List[int]]) -> List[List[int]]:
        """
        Convert arbitrary CNF clauses to 3-SAT format.
        Clauses with more than 3 literals are split using auxiliary variables.
        """
        three_sat_clauses = []

        for clause in clauses:
            if len(clause) <= 3:
                three_sat_clauses.append(clause)
            else:
                # Split clause with more than 3 literals
                # For clause (x1 v x2 v x3 v x4 v ... v xn), introduce auxiliary variables
                # and create: (x1 v x2 v y1) & (~y1 v x3 v y2) & ... & (~yn-3 v xn-1 v xn)
                aux_vars = []
                for i in range(len(clause) - 3):
                    aux_vars.append(self.vpool.id(f"__aux_{self.tseitin_counter}"))
                    self.tseitin_counter += 1

                # First clause: (x1 v x2 v y1)
                three_sat_clauses.append([clause[0], clause[1], aux_vars[0]])

                # Middle clauses: (~yi v x(i+2) v y(i+1))
                for i in range(len(aux_vars) - 1):
                    three_sat_clauses.append(
                        [-aux_vars[i], clause[i + 2], aux_vars[i + 1]]
                    )

                # Last clause: (~yn-3 v xn-1 v xn)
                if len(aux_vars) > 0:
                    three_sat_clauses.append([-aux_vars[-1], clause[-2], clause[-1]])

        return three_sat_clauses


def extract_3sat_from_ast(tree: ast.AST) -> Tuple[List[List[int]], Dict[str, int], int]:
    """
    Extract 3-SAT representation from a transformed AST.

    Returns:
        - List[List[int]]: 3-SAT clauses where sign indicates negation
        - Dict[str, int]: Mapping from variable names to integers
        - int: The variable representing the final output
    """
    # Extract expressions
    extractor = BooleanExpressionExtractor()
    extractor.visit(tree)

    # Convert to CNF
    converter = CNFConverter()

    # Process all assignments
    var_mappings = {}
    for var_name, expr in extractor.expressions:
        var_int = converter.var_to_int(var_name)
        var_mappings[var_name] = var_int
        converter.expr_to_cnf(expr, result_var=var_int)

    # Process the final output
    if extractor.final_output:
        output_var = converter.expr_to_cnf(extractor.final_output)
        # Add unit clause to force output to be true
        converter.clauses.append([output_var])
    else:
        raise ValueError("No return statement found in the function")

    # Convert to 3-SAT
    three_sat_clauses = converter.to_3sat(converter.clauses)

    # Get variable mapping
    var_to_int = {}
    for var_name in converter.vpool.obj2id:
        var_to_int[var_name] = converter.vpool.id(var_name)

    return three_sat_clauses, var_to_int, output_var


def calculate_clause_ratio(clauses: List[List[int]], num_vars: int) -> float:
    """Calculate the clause-to-variable ratio for complexity analysis."""
    return len(clauses) / num_vars if num_vars > 0 else 0
