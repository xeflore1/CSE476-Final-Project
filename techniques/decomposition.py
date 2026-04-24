import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
import ast
from api_wrapper import call_model_chat_completions
load_dotenv()

API_KEY  = os.getenv('API-KEY')
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
    
def decomposition(prompt: str,
                domain: str = "common_sense",
                *,
                model: str = MODEL,
                temperature: float = 0.2,
                max_steps: int = 3,
                max_tokens: int = 512,
                timeout: int = 60,
                **_ignored):
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    max_tokens = 2000
    my_system = "You are a logical assistant. Your job is divide the problem into 3 smaller subproblems whose results can be combined into a solution for the original problem.\n You must answer in UTF-8.\n Each subproblem must be independent of each other (can be solved parallelly) and easy-to-merge with other solutions.\n The output format MUST be EXACTLY:\n [\"subproblem 1\", \"subproblem 2\", \"subproblem 3\"]"
    first_response = calling_api(prompt, my_system, model, temperature, timeout, max_tokens)
    #first_response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    #status = first_response.status_code
    #hdrs   = dict(first_response.headers)
    #with concurrent.futures.ThreadPoolExecutor(max_workers = 3):
    #return first_response
    subproblem_list = ast.literal_eval(first_response["text"])
    new_system = "You are a logical assistant. Think step-by-step and answer the given question. You must answer in UTF-8. Output your answer concisely. Answer MUST be EXACTLY in the format \\boxed{answer}."
    max_tokens = 8000
    print(subproblem_list)
    subq_ans = {}
    subproblem_response = ""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
        futures = {threads.submit(calling_api, subq, new_system, model, temperature, timeout, max_tokens): subq for subq in subproblem_list}
        for i in concurrent.futures.as_completed(futures):
            subq = futures[i]
            try:
                subproblem_response = subproblem_response + "Subproblem:\n" + subq + "Answer:\n" + i.result()["text"] + "\n"

            except Exception as error:
                print("Exception:", error)
    print("Subproblem Response + Context after Completion:", subproblem_response)
    new_system = "You have been provided with 3 subproblems, 3 sub-solutions to those subproblems, and the original problem. Your task is to output a combine solution using each sub-solution provided to you."
    final_prompt = "Original Question: " + prompt + "\n\n" + "The following are 3 subproblems and the corresponding answers to each subproblem:\n" + subproblem_response + "\n\nCombine all of these sub-solutions into a final solution to the question\n" + "Your final answer MUST end with this exact format:\n" + "\\boxed{answer}\n" + "<DONE>"
    max_tokens = 8000
    last_response = calling_api(final_prompt, new_system, model, temperature, timeout, max_tokens)
    return last_response

    




def calling_api(prompt: str, system: str, model: str, temperature: float, timeout: int, max_tokens: int):
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
        "max_tokens": max_tokens,
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


