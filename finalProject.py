import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

API_KEY  = os.getenv('API_KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

def extract_answer(text: str) -> str: # Helper function to extract answer from a models responce.
    if not text:
        return None
    # Extract answer from the format: \\boxed{answer}
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        return match.group(1).strip()
    else:
        return None

# Chain of thought algorithm
def chain_of_thought(prompt: str,
                     system: str = 
                        "You are a logical assistant. Think step-by-step and explain your reasoning clearly before answering." 
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>",
                     model: str = MODEL,
                     temperature: float = 0.3,
                     timeout: int = 120) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    print(f"COT is running with prompt: {prompt}\n")
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 8192,
        "frequency_penalty": 0.0,
        "stop": ["<DONE>"]
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
            verify=False
        )
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            print("200 returned")
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            print("200 is NOT returned")
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

# Self-consistency algorithm
def self_consistency(prompt: str,
                     system: str = (
                        "You are a logical assistant. Think step-by-step and answer the given question."
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>"
                    ),
                     model: str = MODEL,
                     temperature: float = 0.5, # increase temp so that model explores different logical approaches
                     timeout: int = 120) -> dict:

    # Run chain of thought 4 times and pick the most frequent answer
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(chain_of_thought, prompt, system, model, temperature, timeout) for i in range(4)}
        responses = []
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            if response["ok"]:
                answer = extract_answer(response["text"])
                responses.append(answer)
        filteredList = [x for x in responses if x is not None] # get rid of None entries
        if not filteredList:
            return None
        else:
            counts = Counter(filteredList)
            most_common_val = counts.most_common(1)[0][0] # find the most common answer
            return most_common_val # return the most common answer
                

def decomposition(prompt: str,
                   system: str = "You are a helpful assistant. Give the answer to the provided question, but make sure to include all of the logical reasoning steps that you took to arrive at the final answer first.",
                   model: str = MODEL,
                   temperature: float = 0.0,
                   timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

def safe_chain_of_thought(prompt: str):
    result = chain_of_thought(prompt)

    # handle SSL failure / None outputs safely
    if not result.get("ok") or not result.get("text"):
        return {"ok": True, "text": ""}

    return result

def safe_calculator(expression: str) -> str:
    #Evaluate a simple arithmetic expression safely / Supports +, -, *, /, **, and parentheses.\

    import ast
    import operator

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
            raise ValueError("Only numeric constants are allowed.")
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"Unsupported operator: {op_type}")
            return allowed_ops[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return allowed_ops[op_type](_eval(node.operand))
        else:
            raise ValueError("Unsafe expression.")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"CALC_ERROR: {str(e)}"


def tool_augmented_reasoning(prompt: str,
                             system: str = "You are a helpful assistant with access to tools. Decide whether the question needs a calculator. If it does, choose a calculator expression. If it does not, say no tool is needed.",
                             model: str = MODEL,
                             temperature: float = 0.0,
                             timeout: int = 60) -> dict:
    # basic tool reasoning flow:
    # ask model if tool needed
    # if yes -> run calculator
    # then send result back to model
    # otherwise just use CoT
    print(f"Tool-augmented reasoning running with prompt: {prompt}\n")

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    router_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a tool router. Decide if the user question needs a calculator tool.\n"
                    "Return ONLY valid JSON in one of these two forms:\n"
                    "{\"use_tool\": false}\n"
                    "{\"use_tool\": true, \"tool_name\": \"calculator\", \"tool_input\": \"2*(3+4)\"}"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 300,
    }

    try:
        print("Sending router request...")

        router_resp = requests.post(
            url,
            headers=headers,
            json=router_payload,
            timeout=timeout,
            verify=False
        )

        print("Router response received.")

        router_status = router_resp.status_code
        router_hdrs = dict(router_resp.headers)

        print("Status:", router_status)

        if router_status != 200:
            try:
                err_text = router_resp.json()
            except Exception:
                err_text = router_resp.text
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "status": router_status,
                "error": str(err_text),
                "headers": router_hdrs
            }

        router_data = router_resp.json()
        print("Parsed JSON successfully.")

        router_text = router_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        print(f"Router output: {router_text}")

        try:
            #Router - so we decides if it needs a tool or not
            tool_decision = json.loads(router_text)
        except Exception:
            print("Router JSON parse failed, falling back to CoT.")
            return safe_chain_of_thought(prompt)

        if not tool_decision.get("use_tool", False):
            print("No tool needed, falling back to CoT.")
            return safe_chain_of_thought(prompt)

        tool_name = tool_decision.get("tool_name", "")
        tool_input = tool_decision.get("tool_input", "")

        if tool_name != "calculator":
            return {"ok": False, "text": None, "raw": None, "status": -1, "error": f"Unsupported tool: {tool_name}", "headers": {}}

        #Tool execution of calculator
        tool_output = safe_calculator(tool_input)
        print(f"Tool used: {tool_name}")
        print(f"Tool input: {tool_input}")
        print(f"Tool output: {tool_output}")

        #Tool integration back into reasoning
        final_payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the tool result if helpful. "
                        "Give a concise reasoning path and then the final answer. "
                        "Keep your answer less than 5000 characters."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question:\n{prompt}\n\n"
                        f"Tool used: {tool_name}\n"
                        f"Tool input: {tool_input}\n"
                        f"Tool output: {tool_output}\n\n"
                        "Now solve the original question."
                    )
                }
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        final_resp = requests.post(
            url,
            headers=headers,
            json=final_payload,
            timeout=timeout,
            verify=False
        )
        final_status = final_resp.status_code
        final_hdrs = dict(final_resp.headers)

        if final_status == 200:
            final_data = final_resp.json()
            final_text = final_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": final_text, "raw": final_data, "status": final_status, "error": None, "headers": final_hdrs}
        else:
            try:
                err_text = final_resp.json()
            except Exception:
                err_text = final_resp.text
            return {"ok": False, "text": None, "raw": None, "status": final_status, "error": str(err_text), "headers": final_hdrs}

    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

if __name__ == "__main__":
    print("Testing tool augmented reasoning...\n")

    test_question = "What is 25 * 16 + 4?"

    result = tool_augmented_reasoning(test_question)

    print("\nFINAL RESULT:")
    print(result)

def react_agent(prompt: str,
                model: str = MODEL,
                temperature: float = 0.2,
                max_steps: int = 3,
                timeout: int = 60) -> dict:

    print(f"ReAct running with prompt: {prompt}\n")

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    history = []

    for step in range(max_steps):
        print(f"ReAct step {step}")
        # Thought to action to observation to thought to action and finally the observation
        # sidenote model sometimes treats ALL CAPS as emphasis - Gabriel (I'll ask TA/prof if itll work)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning agent that MUST solve problems step-by-step.\n\n"
                    "STRICT RULES:\n"
                    "- You are NOT allowed to solve the problem in one step\n"
                    "- You MUST break the problem into multiple steps\n"
                    "- Each step MUST include Thought and Action\n"
                    "- After each Action, you MUST wait for an Observation\n"
                    "- You MUST explicitly use the Observation in your next Thought\n"
                    "- Do NOT assume results without using the Observation\n"
                    "- Only give Final Answer after multiple steps\n\n"
                    "Format:\n"
                    "Thought: ...\n"
                    "Action: calculator(expression)\n"
                    "OR\n"
                    "Final Answer: ..."
                )
            },
            {"role": "user", "content": prompt}
        ]

        # include past steps
        for h in history:
            messages.append({"role": "assistant", "content": h})

        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512
            },
            timeout=timeout,
            verify=False
        )

        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        print("Model says:", text)

        history.append(text)

        # This is to check if final answer
        if "Final Answer:" in text:
            return {"ok": True, "text": text}

        # so we check if action needed
        match = re.search(r"calculator\((.*?)\)", text)

        if match:
            expr = match.group(1)
            obs = safe_calculator(expr)
            observation = f"Observation: {obs}"
            print(observation)
            history.append(observation)

if __name__ == "__main__":
    print("Testing ReAct...\n")

    test_question = "What is 25 * 16 + 4?"

    result = react_agent(test_question)

    print("\nFINAL RESULT:")
    print(result)
