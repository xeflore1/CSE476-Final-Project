from tools.calculator import calculator
from tools.code_executor import execute_python

TOOL_REGISTRY = {
    "Calculator": calculator,
    "PythonExecutor": execute_python,
    "Finish": lambda x: x
}