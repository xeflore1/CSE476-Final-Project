"""Agent router / orchestrator. Entry point: agent(prompt)."""

from domain_classifier import classify_domain
from techniques.prompt_optimization import prompt_optimized_call
from techniques.llm_as_judge import confidence_check

from techniques.chain_of_thought import chain_of_thought
from techniques.self_consistency import self_consistency
from techniques.tree_of_thought import tree_of_thought
from techniques.self_refine import self_refine
from techniques.react_agent import react_agent
from techniques.tool_augmented import tool_augmented
from techniques.decomposition import decomposition
from techniques.ensemble_voting import ensemble_vote
from api_wrapper import MODEL

TECHNIQUES = {
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
    "tree_of_thought": tree_of_thought,
    "self_refine": self_refine,
    "react": react_agent,
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
ENSEMBLE_COST = {
    "math": 8,
    "common_sense": 8,
    "coding": 6,
    "future_prediction": 9,
    "planning": 10,
}
SAFETY_MARGIN = 1


def _run_counted(fn, *args, **kwargs):
    # Call a technique and return (answer, calls_used, full_result).
    res = fn(*args, **kwargs)
    return res.get("answer", ""), res.get("calls", 0), res


_TOK = {"math": 256, "common_sense": 256, "coding": 1024, "future_prediction": 256, "planning": 512}


def agent(prompt: str, *, verbose: bool = False) -> str:
    try:
        calls_used = 0
        domain = classify_domain(prompt)
        opt = prompt_optimized_call(
            prompt,
            domain,
            MODEL,
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
        if calls_used < BUDGET_PER_QUESTION - 3 and ans:
            confidence = confidence_check(prompt, ans)
            calls_used += confidence.get("calls", 0)
            if confidence.get("level") == "low":
                remaining = BUDGET_PER_QUESTION - calls_used
                need = ENSEMBLE_COST.get(domain, 8) + SAFETY_MARGIN
                if remaining >= need:
                    ens = ensemble_vote(
                        optimized,
                        domain,
                        techniques_dict=TECHNIQUES,
                        budget=remaining - SAFETY_MARGIN,
                        max_tokens=_TOK.get(domain, 1024),
                        timeout=180,
                    )
                    calls_used += ens.get("calls", 0)
                    if ens.get("ok") and ens.get("answer"):
                        ans = ens["answer"]
                elif remaining >= 4:
                    ens = ensemble_vote(
                        optimized,
                        domain,
                        techniques_dict=TECHNIQUES,
                        budget=remaining - SAFETY_MARGIN,
                        max_tokens=_TOK.get(domain, 1024),
                        timeout=180,
                    )
                    calls_used += ens.get("calls", 0)
                    if ens.get("ok") and ens.get("answer"):
                        ans = ens["answer"]
                elif remaining >= 3:
                    sr = self_refine(optimized, domain, max_iterations=1)
                    calls_used += sr.get("calls", 0)
                    if sr.get("ok") and sr.get("answer"):
                        ans = sr["answer"]
        if verbose:
            print(f"[router] final_calls={calls_used}")
        if ans is None:
            ans = ""
        if len(ans) > 4900:
            ans = ans[:4900]
        return ans
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is 2 + 2?"
    print(agent(q, verbose=True))
