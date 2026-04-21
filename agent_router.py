from api_wrapper import MODEL
from domain_classifier import classify_domain
from answer_extraction import extract_answer
from techniques.prompt_optimization import optimize_prompt
from techniques.llm_as_judge import confidence_check

# Technique imports
from techniques.chain_of_thought import chain_of_thought
from techniques.self_consistency import self_consistency
from techniques.tree_of_thought import tree_of_thought
from techniques.self_refine import self_refine
from techniques.react_agent import react
from techniques.tool_augmented import tool_augmented
from techniques.decomposition import decomposition
from techniques.ensemble_voting import ensemble_vote
import finalProject as final_project

_TOK = {"math": 256, "common_sense": 256, "coding": 1024, "future_prediction": 256, "planning": 512}

_LLM_ESTIMATE = {
    "prompt_optimized_call": 1,
    "chain_of_thought": 1,
    "self_consistency": 4,
    "tree_of_thought": 8,
    "self_refine": 3,
    "react": 4,
    "tool_augmented": 3,
    "decomposition": 5,
    "ensemble_vote": 6,
}


class QuestionLLMBudget:
    __slots__ = ("used", "limit")

    def __init__(self, limit=18):
        self.used = 0
        self.limit = limit

    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def estimate_technique_llm_calls(technique) -> int:
    return _LLM_ESTIMATE.get(getattr(technique, "__name__", ""), 1)


def track_llm_calls(budget: QuestionLLMBudget, technique) -> bool:
    n = estimate_technique_llm_calls(technique)
    if budget.used + n > budget.limit:
        return False
    budget.used += n
    return True


def _normalize_raw(res):
    if isinstance(res, dict):
        if not res.get("ok"):
            return ""
        t = res.get("text")
        return t if t is not None else ""
    if res is None:
        return ""
    return str(res)


def _invoke(technique, question_text, domain):
    if getattr(technique, "__name__", "") == "prompt_optimized_call":
        return prompt_optimized_call(
            question_text,
            domain,
            final_project.MODEL,
            0.0,
            _TOK.get(domain, 256),
            180,
        )
    return technique(question_text)


def route_question(question_text, domain, budget=None):
    techniques = []
    if domain == "math":
        techniques.append(chain_of_thought)
        techniques.append(self_consistency)
    elif domain == "common_sense":
        techniques.append(prompt_optimized_call)
    elif domain == "coding":
        techniques.append(react)
        techniques.append(tool_augmented)
    elif domain == "future_prediction":
        techniques.append(prompt_optimized_call)
    elif domain == "planning":
        techniques.append(decomposition)
    if budget is None:
        return techniques
    rem = budget.remaining()
    return [t for t in techniques if estimate_technique_llm_calls(t) <= rem]


def agent(question_text) -> str:
    domain = classify_domain(question_text)
    budget = QuestionLLMBudget()
    techniques = route_question(question_text, domain, budget)
    for technique in techniques:
        if not track_llm_calls(budget, technique):
            continue
        raw = _invoke(technique, question_text, domain)
        text = _normalize_raw(raw)
        out = extract_answer(text, domain)
        if out:
            return out
    return ""
