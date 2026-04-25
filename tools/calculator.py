import ast
import operator

# all of the allowed operations
allowed_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric values are allowed.")

    elif isinstance(node, ast.Num):
        return node.n

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in allowed_ops:
            raise ValueError("Unsupported operator")
        return allowed_ops[op_type](_eval(node.left), _eval(node.right))

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in allowed_ops:
            raise ValueError("Unsupported unary operator")
        return allowed_ops[op_type](_eval(node.operand))

    else:
        raise ValueError("Expression is not allowed")

def calculator(expression: str) -> str:
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"CALC_ERROR: {str(e)}"