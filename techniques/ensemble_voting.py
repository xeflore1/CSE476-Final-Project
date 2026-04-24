import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
import ast
from techniques.self_refine import self_refinement
from techniques.tree_of_thought import tree_of_thought
from techniques.chain_of_thought import chain_of_thought
from techniques.decomposition import decomposition
from techniques.tool_augmented import tool_augmented
from techniques.prompt_optimization import prompt_optimized_call
from techniques.self_consistency import self_consistency
from techniques.react_agent import react_agent
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

DOMAIN_TO_TECHNIQUES = {
    "math": [chain_of_thought, decomposition, tool_augmented],
    "common_sense": [chain_of_thought, self_refinement, prompt_optimized_call],
    "coding": [react_agent, chain_of_thought, self_refinement],
    "future_prediction": [chain_of_thought, self_consistency, prompt_optimized_call],
    "planning": [decomposition, chain_of_thought, tree_of_thought]
}
def extract_answer(text: str) -> str: # Helper function to extract answer from a models responce.
    if not text:
        return None
    # Extract answer from the format: \\boxed{answer}
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        return match.group(1).strip()
    else:
        return None
        #return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

def ensemble_vote(prompt: str,
                   domain: str = "common_sense",
                   *,
                    techniques_dict: dict | None = None,
                  max_tokens: int = 1024,
                  timeout: int = 120,
                  **_ignored) -> dict:
    answers_list = []
    max_tokens = 5000
    techniques_to_be_used = DOMAIN_TO_TECHNIQUES[domain]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
        thread = [threads.submit(DOMAIN_TO_TECHNIQUES[i], prompt, domain, max_tokens, timeout) for i in techniques_to_be_used]
        #Process threads
        answers_list = [a.result() for a in thread]   
    if domain == "planning":
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