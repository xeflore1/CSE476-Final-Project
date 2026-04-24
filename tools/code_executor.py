#so calc does math; however, this is for logic loops that are needed in ReAct

def execute_python(code: str) -> str:
    try:
        local_env = {}
        exec(code, {"__builtins__": {}}, local_env)
        return str(local_env)
    except Exception as e:
        return f"EXEC_ERROR: {str(e)}"