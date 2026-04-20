import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
import ast
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
                   system: str = "You are a logical assistant. Your job is divide the problem into 3 smaller subproblems whose results can be combined into a solution for the original problem.\n You must answer in UTF-8.\n Each subproblem must be independent of each other (can be solved parallelly) and easy-to-merge with other solutions.\n The output format MUST be EXACTLY:\n [\"subproblem 1\", \"subproblem 2\", \"subproblem 3\"]",
                   model: str = MODEL,
                   temperature: float = 0.15,
                   timeout: int = 180):
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    max_tokens = 300
    first_response = calling_api(prompt, system, model, temperature, timeout, max_tokens)
    #first_response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    #status = first_response.status_code
    #hdrs   = dict(first_response.headers)
    #with concurrent.futures.ThreadPoolExecutor(max_workers = 3):
    #return first_response
    subproblem_list = ast.literal_eval(first_response["text"])
    new_system = "You are a logical assistant. Think step-by-step and answer the given question. You must answer in UTF-8"
    max_tokens = 5000
    #print(subproblem_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
        futures = {threads.submit(calling_api, subproblem, new_system, model, temperature, timeout, max_tokens) for subproblem in subproblem_list}
        subproblem_response = ""
        for i in concurrent.futures.as_completed(futures):
            try:
                subproblem_response = subproblem_response + i.result()["text"] + "\n"

            except Exception as error:
                print("Exception:", error)
    final_prompt = "Question: " + prompt + "\n" + "The following 3 answers are the answers to each subproblem:\n" + subproblem_response + "\n\nCombine all of these sub-solutions into a final solution to the question\n" + "Your final answer MUST end with this exact format:\n" + "\\boxed{answer}\n" + "<DONE>"
    max_tokens = 5000
    last_response = calling_api(final_prompt, new_system, model, temperature, timeout, max_tokens)
    return extract_answer(last_response["text"])

    




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

def ensemble_voting(prompt: str,
                   system: str = "You are a logical assistant. Your job is to classify the problem into EXACTLY one of the following domains: math, common_sense, coding, future_prediction, or planning.\nYour final answer MUST end with this exact format:\n" + "\\boxed{answer}\n" + "<DONE>",
                   model: str = MODEL,
                   temperature: float = 0.15,
                   timeout: int = 180):
    # First have to find domain of problem
    first_call = calling_api(prompt, system, model, temperature, timeout, max_tokens=500)
    domain = extract_answer(first_call["text"])
    answers_list = []
    if domain == "math":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, new_system, model, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(decomposition, prompt, new_system, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(tool_augmented_reasoning, prompt, new_system, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread)
            answers_list.append(decomp_thread)
            answers_list.append(tool_aug_thread)
        answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return "No answer could be found"
        else:
            counter_object = Counter(answers_list)
            return counter_object.most_common()[0][0]
        
    elif domain == "common_sense":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, new_system, model, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(self_refine, prompt, new_system, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(prompt_optimization, prompt, new_system, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread)
            answers_list.append(decomp_thread)
            answers_list.append(tool_aug_thread)
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return "No answer could be found"
        else:
            counter_object = Counter(answers_list)
            return counter_object.most_common()[0][0]
        
    elif domain == "coding":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(react_agent, prompt, new_system, model, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(chain_of_thought, prompt, new_system, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(self_refine, prompt, new_system, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread)
            answers_list.append(decomp_thread)
            answers_list.append(tool_aug_thread)
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return "No answer could be found"
        else:
            counter_object = Counter(answers_list)
            return counter_object.most_common()[0][0]
        
    elif domain == "future_prediction":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, new_system, model, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(self_consistency, prompt, new_system, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(prompt_optimization, prompt, new_system, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread)
            answers_list.append(decomp_thread)
            answers_list.append(tool_aug_thread)
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return "No answer could be found"
        else:
            counter_object = Counter(answers_list)
            return counter_object.most_common()[0][0]
        
    elif domain == "planning":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(decomposition, prompt, new_system, model, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(chain_of_thought, prompt, new_system, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(tree_og_thought, prompt, new_system, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread)
            answers_list.append(decomp_thread)
            answers_list.append(tool_aug_thread)
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return "No answer could be found"
        else:
            counter_object = Counter(answers_list)
            return counter_object.most_common()[0][0]
