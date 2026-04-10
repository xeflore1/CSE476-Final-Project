import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")


def helloworld():
    print("Hello, world!")

def extract_answer(text: str) -> str:
    # Helper to pull the final answer from the CoT text.
    print(f"extracting answer from: {text}")
    if not text:
        return None
    # retrieves answer from the end of the response
    match = re.search(r"Final Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip('.')
    return None

# Chain of thought algorithm
def chain_of_thought(prompt: str,
                     system: str = "You are a logical assistant. Think step-by-step and explain your reasoning clearly before answering.", # You should provide the final answer in the format: \n\nFinal Answer: [answer]\n\n",
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
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
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


def self_consistency(prompt: str,
                     system: str = (
                        "You are a logical assistant. First, think step-by-step and explain your reasoning clearly. "
                        "Once you have completed your reasoning, you MUST output your final answer strictly using the following format:\n\n"
                        "Final Answer:\n"
                        "[Your final answer goes here]\n"
                        "<DONE>"
                    ),
                     model: str = MODEL,
                     temperature: float = 0.5, # increase temp so that model explores different logical approaches
                     timeout: int = 120) -> dict:
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(chain_of_thought, prompt, system, model, temperature, timeout) for i in range(3)}
        responses = []
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            if response["ok"]:
                answer = extract_answer(response["text"])
                responses.append(answer)
        print(f"Responses: {responses}")
        counts = Counter(responses)
        most_common_val = counts.most_common(1)[0][0]
        return most_common_val
                

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
