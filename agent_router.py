from domain_classifier import classify_domain
from techniques.prompt_optimization import prompt_optimized_call
from techniques.llm_as_judge import confidence_check

from techniques.chain_of_thought import chain_of_thought
from techniques.self_consistency import self_consistency
from techniques.tree_of_thought import tree_of_thought
from techniques.self_refine import self_refine
from techniques.react_agent import react
from techniques.tool_augmented import tool_augmented
from techniques.decomposition import decomposition
from techniques.ensemble_voting import ensemble_vote
import finalProject as final_project

TECHNIQUES = {
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
    "tree_of_thought": tree_of_thought,
    "self_refine": self_refine,
    "react": react,
    "tool_augmented": tool_augmented,
    "decomposition": decomposition,
}

_DEFAULT_FIRST = {
    "math": "chain_of_thought",
    "coding": "tool_augmented",
    "common_sense": "chain_of_thought",
    "planning": "tree_of_thought",
    "future_prediction": "self_refine",
}

BUDGET_PER_QUESTION = 20


def _run_counted(fn, *args, **kwargs):
    # Call a technique and return (answer, calls_used, full_result).
    res = fn(*args, **kwargs)
    return res.get("answer", ""), res.get("calls", 0), res


_TOK = {"math": 256, "common_sense": 256, "coding": 1024, "future_prediction": 256, "planning": 512}


def agent(prompt: str, *, verbose: bool = False) -> str:
    calls_used = 0
    domain = classify_domain(prompt)
    opt = prompt_optimized_call(
        prompt,
        domain,
        final_project.MODEL,
        0.0,
        _TOK.get(domain, 256),
        180,
    )
    calls_used += opt.get("calls", 0)
    optimized = opt.get("optimized") or prompt
    primary_name = _DEFAULT_FIRST.get(domain, "chain_of_thought")
    primary_fn = TECHNIQUES[primary_name]
    ans, c, _ = _run_counted(
        primary_fn,
        optimized,
        domain,
        max_tokens=min(1024, 50 * max(1, BUDGET_PER_QUESTION - calls_used)),
    )
    calls_used += c
    if verbose:
        print(f"[router] domain={domain} primary={primary_name} calls_so_far={calls_used}")
    if ans is None:
        ans = ""
    if len(ans) > 4900:
        ans = ans[:4900]
    return ans
