import re
from tools import TOOL_REGISTRY
from api_wrapper import call_model_chat_completions

TOOL_PATTERN = re.compile(r"(Calculator|PythonExecutor)\[(.*?)\]")

def tool_augmented(prompt: str, domain: str = "common_sense", *, max_tokens: int = 1024, timeout: int = 120, **_ignored) -> dict:
    # basic tool reasoning flow:
    # ask model whether a tool is needed
    # if so, run the tool and pass result back
    # otherwise return the model response directly
    print(f"Tool-augmented running with prompt: {prompt}\n")

    # ask model what to do
    response = call_model_chat_completions(
        messages=[
            {
                "role": "system",
                "content": (
                    "You can use tools.\n"
                    "If needed, respond EXACTLY in this format:\n"
                    "Calculator[expression]\n"
                    "or\n"
                    "PythonExecutor[code]\n"
                    "Otherwise respond with:\n"
                    #"Final Answer: ..."
                    #Proper identation
                    "Action: Finish[answer]"
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        timeout=timeout
    )

    if not response["ok"]:
        return response

    text = response.get("text", "")
    print("Model output:", text)

    # check if tool needed
    match = TOOL_PATTERN.search(text)
    finish = re.search(r"Action:\s*Finish\[(.*?)\]", text, re.DOTALL)

    if finish and not match:
        return {
            "ok": True,
            "text": text,
            "answer": finish.group(1).strip(),
            "calls": 1,
            "error": None
        }

    if not match:
        # no tool needed
        return {
            "ok": True,
            "text": text,
            "answer": text,
            "calls": 1,
            "error": None
        }

    tool_name, tool_input = match.groups()

    print(f"Tool: {tool_name}")
    print(f"Input: {tool_input}")

    # run tool
    try:
        tool_fn = TOOL_REGISTRY[tool_name]
        tool_output = tool_fn(tool_input)
    except Exception as e:
        return {
            "ok": False,
            "text": None,
            "answer": "",
            "calls": 1,
            "error": str(e)
        }

    print("Tool output:", tool_output)

    # send result back to model
    final = call_model_chat_completions(
        messages=[
            {
                "role": "system",
                "content": "Use the tool result to answer the question."
            },
            {
                "role": "user",
                "content": (
                    f"Question: {prompt}\n"
                    f"Tool output: {tool_output}\n"
                    "Now give the final answer."
                )
            }
        ],
        max_tokens=max_tokens,
        timeout=timeout
    )

    final_text = final.get("text") or ""
    final_finish = re.search(r"Action:\s*Finish\[(.*?)\]", final_text, re.DOTALL)
    final_answer = final_finish.group(1).strip() if final_finish else final_text

    return {
        "ok": final["ok"],
        "text": final_text,
        "answer": final_answer,
        "calls": 2,
        "error": final.get("error")
    }
