"""Agent router / orchestrator. Entry point: agent(prompt)."""

import re

from domain_classifier import classify_domain
from techniques.prompt_optimization import prompt_optimized_call
from techniques.llm_as_judge import confidence_check
from techniques.output_instructions import output_instructions

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
    "coding": "chain_of_thought",
    "common_sense": "chain_of_thought",
    "planning": "chain_of_thought",
    "future_prediction": "chain_of_thought",
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
    res = fn(*args, **kwargs)
    return res.get("answer", ""), res.get("calls", 0), res


_TOK = {"math": 8192, "common_sense": 8192, "coding": 8192, "future_prediction": 8192, "planning": 8192}

_SKIP_OPTIMIZER = {"planning", "future_prediction", "coding"}

LONG_PROMPT_CHARS = 6000
VERY_LONG_PROMPT_CHARS = 16000


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ACTION_LINE = re.compile(r"^\s*\([a-zA-Z_][\w\-]*(?:\s+[\w\-]+)*\s*\)\s*$")


def _postprocess_answer(domain: str, ans: str) -> str:
    if not ans:
        return ans
    if domain == "planning":
        m = re.search(r"\[PLAN\]\s*(.*?)\s*\[PLAN END\]", ans, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
        action_lines = [ln.strip() for ln in ans.splitlines() if _ACTION_LINE.match(ln)]
        if action_lines:
            return "\n".join(action_lines)
        return ans
    if domain == "coding":
        m = _CODE_FENCE.search(ans)
        if m and m.group(1).strip():
            return m.group(1).rstrip()
        return ans
    return ans


def _resolve_max_tokens(domain: str, prompt: str) -> int:
    base = _TOK.get(domain, 1024)
    if len(prompt) >= VERY_LONG_PROMPT_CHARS:
        return min(base, 384)
    if len(prompt) >= LONG_PROMPT_CHARS:
        return min(base, 512)
    return base


def _rescue_answer(prompt: str, domain: str, calls_used: int, verbose: bool):
    #Try simpler / cleaner techniques when the primary returned nothing. Returns (answer, additional_calls_used).
    extra_calls = 0

    if calls_used + 1 <= BUDGET_PER_QUESTION:
        try:
            print(f"[TECHNIQUE] rescue: chain_of_thought (clean)  domain={domain}")
            r1 = chain_of_thought(
                prompt,
                domain,
                max_tokens=_resolve_max_tokens(domain, prompt),
                system=output_instructions(domain),
                temperature=0.0,
            )
            extra_calls += r1.get("calls", 0)
            ans1 = r1.get("answer") or ""
            if verbose:
                print(f"[router] rescue#1 cot_clean calls={r1.get('calls', 0)} ok={r1.get('ok')} len={len(ans1)}")
            if ans1.strip():
                return ans1, extra_calls
        except Exception as e:
            if verbose:
                print(f"[router] rescue#1 exception: {e}")

    if calls_used + extra_calls + 2 <= BUDGET_PER_QUESTION:
        try:
            print(f"[TECHNIQUE] rescue: self_refine  domain={domain}")
            r2 = self_refine(prompt, domain, max_iterations=1, max_tokens=_resolve_max_tokens(domain, prompt))
            extra_calls += r2.get("calls", 0)
            ans2 = r2.get("answer") or ""
            if verbose:
                print(f"[router] rescue#2 self_refine calls={r2.get('calls', 0)} ok={r2.get('ok')} len={len(ans2)}")
            if ans2.strip():
                return ans2, extra_calls
        except Exception as e:
            if verbose:
                print(f"[router] rescue#2 exception: {e}")

    return "", extra_calls


def agent(prompt: str, *, verbose: bool = False) -> str:
    try:
        calls_used = 0
        domain = classify_domain(prompt)
        is_long = len(prompt) >= LONG_PROMPT_CHARS
        skip_optimizer = (domain in _SKIP_OPTIMIZER) or is_long
        if skip_optimizer:
            optimized = prompt
            if verbose and is_long:
                print(f"[router] long prompt ({len(prompt)} chars) -> skip optimizer")
        else:
            print(f"[TECHNIQUE] prompt_optimization  domain={domain}")
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
        primary_max_tokens = _resolve_max_tokens(domain, optimized)
        print(f"[TECHNIQUE] {primary_name}  domain={domain}")
        ans, c, _ = _run_counted(
            primary_fn,
            optimized,
            domain,
            max_tokens=primary_max_tokens,
            system=output_instructions(domain),
        )
        calls_used += c
        if verbose:
            print(f"[router] domain={domain} primary={primary_name} calls_so_far={calls_used} ans_len={len(ans or '')}")

        if not (ans or "").strip():
            rescue_ans, rescue_calls = _rescue_answer(prompt, domain, calls_used, verbose)
            calls_used += rescue_calls
            if rescue_ans:
                ans = rescue_ans

        if calls_used < BUDGET_PER_QUESTION - 3 and (ans or "").strip():
            print(f"[TECHNIQUE] llm_as_judge (confidence_check)  domain={domain}")
            confidence = confidence_check(prompt, ans)
            calls_used += confidence.get("calls", 0)
            if confidence.get("level") == "low":
                remaining = BUDGET_PER_QUESTION - calls_used
                need = ENSEMBLE_COST.get(domain, 8) + SAFETY_MARGIN
                if remaining >= need:
                    print(f"[TECHNIQUE] ensemble_voting  domain={domain} budget={remaining - SAFETY_MARGIN}")
                    ens = ensemble_vote(
                        optimized,
                        domain,
                        techniques_dict=TECHNIQUES,
                        budget=remaining - SAFETY_MARGIN,
                        max_tokens=primary_max_tokens,
                        timeout=180,
                    )
                    calls_used += ens.get("calls", 0)
                    if ens.get("ok") and ens.get("answer"):
                        ans = ens["answer"]
                elif remaining >= 4:
                    print(f"[TECHNIQUE] ensemble_voting  domain={domain} budget={remaining - SAFETY_MARGIN}")
                    ens = ensemble_vote(
                        optimized,
                        domain,
                        techniques_dict=TECHNIQUES,
                        budget=remaining - SAFETY_MARGIN,
                        max_tokens=primary_max_tokens,
                        timeout=180,
                    )
                    calls_used += ens.get("calls", 0)
                    if ens.get("ok") and ens.get("answer"):
                        ans = ens["answer"]
                elif remaining >= 3:
                    print(f"[TECHNIQUE] self_refine  domain={domain} remaining={remaining}")
                    sr = self_refine(optimized, domain, max_iterations=1)
                    calls_used += sr.get("calls", 0)
                    if sr.get("ok") and sr.get("answer"):
                        ans = sr["answer"]
        if verbose:
            print(f"[router] final_calls={calls_used} final_len={len(ans or '')}")
        if ans is None:
            ans = ""
        ans = _postprocess_answer(domain, ans)
        if len(ans) > 4900:
            ans = ans[:4900]
        return ans
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is 2 + 2?"
    print(agent(q, verbose=True))
