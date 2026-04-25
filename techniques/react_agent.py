import re
from tools import TOOL_REGISTRY
from api_wrapper import call_model_chat_completions, MODEL

def react_agent(prompt: str,
                model: str = MODEL,
                temperature: float = 0.2,
                max_steps: int = 3,
                timeout: int = 60) -> dict:

    print(f"ReAct running with prompt: {prompt}\n")

    history = []
    calls = 0

    for step in range(max_steps):
        print(f"ReAct step {step}")
        # build messages for this step of the react loop
        # model sometimes treats ALL CAPS as emphasis - Gabriel (I'll ask TA/prof if itll work)

        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the problems step by step.\n"
                    "Use multiple steps when needed.\n"
                    "Each step should have a Thought AND an Action.\n"
                    "After an Action, please wait for an Observation before continuing.\n"
                    "Do NOT assume results without using the Observation.\n"
                    "Only give the answer after multiple steps have occured.\n\n"
                    "Format:\n"
                    "Thought: ...\n"
                    "Action: Calculator[expression]\n"
                    "or\n"
                    "Action: Finish[answer]"
                )
            },
            {"role": "user", "content": prompt}
        ]

        # include past steps
        for h in history:
            messages.append(h)

        resp = call_model_chat_completions(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=512,
            timeout=timeout
        )

        calls += resp.get("calls", 1)

        if not resp["ok"]:
            return {
                "ok": False,
                "text": None,
                "answer": "",
                "calls": calls,
                "error": resp.get("error")
            }

        text = resp.get("text", "")

        print("Model says:", text)

        history.append({"role": "assistant", "content": text})

        # This is to check if final answer
        if "Action: Finish[" in text:
            answer = re.search(r"Finish\[(.*?)\]", text)
            return {
                "ok": True,
                "text": text,
                "answer": answer.group(1) if answer else "",
                "calls": calls,
                "error": None
            }

        # so we check if action needed
        match = re.search(r"Calculator\[(.*?)\]", text)

        if match:
            expr = match.group(1)

            try:
                obs = TOOL_REGISTRY["Calculator"](expr)
            except Exception as e:
                obs = f"ERROR: {str(e)}"

            observation = f"Observation: {obs}"
            print(observation)


            history.append({"role": "user", "content": observation})

    return {
        "ok": True,
        "text": text,
        #"answer": "",
        "answer": text,
        "calls": calls,
        "error": None
    }