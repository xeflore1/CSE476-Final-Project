import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter, OrderedDict
from dotenv import load_dotenv
import ast
from techniques.self_refine import self_refine
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
    "math": ["chain_of_thought", "decomposition", "tool_augmented"],
    "common_sense": ["chain_of_thought", "self_refine", "self_consistency"],
    "coding": ["react_agent", "chain_of_thought", "self_refine"],
    "future_prediction": ["chain_of_thought", "self_consistency", "self_consistency"],
    "planning": ["decomposition", "chain_of_thought", "tree_of_thought"]
}

techniques_call_cost = {
    "chain_of_thought": 1, "tool_augmented": 2, "self_refine": 3, "react_agent": 3, "self_consistency": 4, "tree_of_thought": 4, "decomposition": 5
}

def plan_for_budget(domain: str, budget: int) -> list[str] | None:
    ideal = DOMAIN_TO_TECHNIQUES.get(domain, DOMAIN_TO_TECHNIQUES["common_sense"])
    ideal_cost = sum(techniques_call_cost.get(i, 99) for i in ideal)
    if ideal_cost <= budget:
        return ideal

    # fallback: pick the cheapest techniques that fit
    sorted_techs = sorted(techniques_call_cost.items(), key=lambda kv: kv[1])
    acc, picked = 0, []
    for name, cost in sorted_techs:
        if acc + cost > budget:
            continue
        acc += cost
        picked.append(name)
        if len(picked) >= 3:
            break
    return picked or None
    


def ensemble_vote(prompt: str,
                   domain: str = "common_sense",
                   *,
                    techniques_dict: dict | None = None,
                    budget: int = 4,
                  max_tokens: int = 1024,
                  timeout: int = 120,
                  **_ignored) -> dict:
    print("Running Ensemble Voting Now")
    techniques_dict = techniques_dict or {}
    budget_calls = plan_for_budget(domain, budget)
    tech_functions = [techniques_dict[n] for n in (budget_calls or []) if n in techniques_dict]
    if not tech_functions:
        return {
        "ok": False,
        "text": None,
        "answer": "",
        "calls": 0,
        "error": "no usable techniques in techniques_dict"
    }
    answers_list = []
    max_tokens = 5000
    # Finally running threads of each technique
    calls = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
        thread = {threads.submit(func, prompt=prompt, domain=domain, max_tokens=max_tokens, timeout=timeout): func for func in tech_functions}
        #Process threads
        for a in thread:
            try:
                #answers_list.append(a.result())
                tech_answer = a.result()
                calls += tech_answer.get("calls", 0)
                if tech_answer.get("ok"):
                    answers_list.append(tech_answer.get("answer") or tech_answer.get("text"))
                else:
                    answers_list.append(tech_answer.get("error"))
            except Exception as e:
                print(f"Answer Failed in Ensemble Voting: {e}")

    if not answers_list:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": None, "headers": {}}
    else:
        answer = Counter(answers_list).most_common(1)[0][0]
        return {"ok": True, "text": answer, "answer": answer, "calls": calls, "error": None}
    