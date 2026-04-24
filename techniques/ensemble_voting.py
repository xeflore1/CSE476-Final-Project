import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
import ast
from self_refine import self_refine
from tree_of_thought import tree_of_thought
from chain_of_thought import chain_of_thought
from decomposition import decomposition
from tool_augmented import tool_augmented
from prompt_optimization import prompt_optimized_call
from self_consistency import self_consistency
from react_agent import react_agent
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

def ensemble_vote(prompt: str,
                   system: str = "You are a logical assistant. Your job is to classify the problem into EXACTLY one of the following domains: math, common_sense, coding, future_prediction, or planning.\nYour final answer MUST end with this exact format:\n" + "\\boxed{answer}\n" + "<DONE>",
                   model: str = MODEL,
                   temperature: float = 0.15,
                   timeout: int = 180):
    # First have to find domain of problem
    first_call = calling_api(prompt, system, model, temperature, timeout, max_tokens=500)
    domain = extract_answer(first_call["text"])
    answers_list = []
    max_tokens = 5000
    if domain == "math":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, temperature, timeout, max_tokens)
            decomp_thread = threads.submit(decomposition, prompt, model, temperature, timeout, max_tokens)
            tool_aug_thread = threads.submit(tool_augmented, prompt, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread.result())
            answers_list.append(decomp_thread.result())
            answers_list.append(tool_aug_thread.result())
        answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return {'ok': False, 'text': "No answer could be found"}
        else:
            counter_object = Counter(answers_list)
            return {'ok': True, "text": counter_object.most_common()[0][0]}
        
    elif domain == "common_sense":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, model, temperature, timeout, max_tokens)
            self_refine_thread = threads.submit(self_refinement, prompt, model, temperature, timeout, max_tokens)
            prompt_opt_thread = threads.submit(prompt_optimized_call, prompt,  model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread.result())
            answers_list.append(self_refine_thread.result())
            answers_list.append(prompt_opt_thread.result())
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return {'ok': False, 'text': "No answer could be found"}
        else:
            counter_object = Counter(answers_list)
            return {'ok': True, 'text': counter_object.most_common()[0][0]}
        
    elif domain == "coding":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            react_thread = threads.submit(react_agent, prompt, model, temperature, timeout, max_tokens)
            cot_thread = threads.submit(chain_of_thought, prompt, model, temperature, timeout, max_tokens)
            self_refine_thread = threads.submit(self_refinement, prompt, model, temperature, timeout, max_tokens)

            answers_list.append(react_thread.result())
            answers_list.append(cot_thread.result())
            answers_list.append(self_refine_thread.result())
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return {'ok': False, 'text': "No answer could be found"}
        else:
            counter_object = Counter(answers_list)
            return {'ok': True, 'text': counter_object.most_common()[0][0]}
        
    elif domain == "future_prediction":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            cot_thread = threads.submit(chain_of_thought, prompt, model, temperature, timeout, max_tokens)
            self_con_thread = threads.submit(self_consistency, prompt, model, temperature, timeout, max_tokens)
            prompt_opt_thread = threads.submit(prompt_optimized_call, prompt, model, temperature, timeout, max_tokens)

            answers_list.append(cot_thread.result())
            answers_list.append(self_con_thread.result())
            answers_list.append(prompt_opt_thread.result())
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return {'ok': False, 'text': "No answer could be found"}
        else:
            counter_object = Counter(answers_list)
            return {'ok': True, 'text': counter_object.most_common()[0][0]}
        
    elif domain == "planning":
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            decomp_thread = threads.submit(decomposition, prompt, model, temperature, timeout, max_tokens)
            cot_thread = threads.submit(chain_of_thought, prompt, model, temperature, timeout, max_tokens)
            tree_thread = threads.submit(tree_of_thought, prompt, model, temperature, timeout, max_tokens)

            answers_list.append(decomp_thread.result())
            answers_list.append(cot_thread.result())
            answers_list.append(tree_thread.result())
            answers_list = [x for x in answers_list if x is not None]
        if not answers_list:
            return {'ok': False, 'text': "No answer could be found"}
        else:
            counter_object = Counter(answers_list)
            return {'ok': True, 'text': counter_object.most_common()[0][0]}