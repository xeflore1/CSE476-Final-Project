import builtins
import contextlib
import io

# so calc does math; however, this is for logic loops that are needed in ReAct
_SAFE_BUILTINS = {
    name: getattr(builtins, name) for name in (
        "print", "range", "len", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "str", "int", "float", "bool", "list", "dict", "tuple", "set",
        "any", "all", "isinstance"
    )
}


def execute_python(code: str) -> str:
    buf = io.StringIO()
    local_env: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": _SAFE_BUILTINS}, local_env)
        out = buf.getvalue().strip()
        return out if out else str({k: v for k, v in local_env.items() if not k.startswith("_")})
    except Exception as e:
        return f"EXEC_ERROR: {type(e).__name__}: {e}"
#so calc does math; however, this is for logic loops that are needed in ReAct

def execute_python(code: str) -> str:
    try:
        local_env = {}
        exec(code, {"__builtins__": {}}, local_env)
        return str(local_env)
    except Exception as e:
        return f"EXEC_ERROR: {str(e)}"