from __future__ import annotations
import re
from api_wrapper import call_model_chat_completions, MODEL

JUDGE_MAX_TOKENS = 32


def _chat(system, user, model, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": JUDGE_MAX_TOKENS,
    }
    try:
        res = call_model_chat_completions(payload, model=model, temperature=temperature, timeout=120)
        if not res["ok"]:
            return {"ok": False, "text": None}
        return {"ok": True, "text": res["text"]}
    except Exception as e:
        return {"ok": False, "text": None, "error": str(e)}

# Helper function to parse the boolean response from the LLM.
def _parse_bool(text: str | None, prediction, expected) -> bool:
    fallback = str(prediction).strip() == str(expected).strip()
    if not text:
        return fallback
    t = text.strip()
    if re.search(r"\bTrue\b", t):
        return True
    if re.search(r"\bFalse\b", t):
        return False
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return fallback

# Helper function to parse the choice response from the LLM.
def _parse_choice(text: str | None, n: int) -> int:
    if not text or n < 1:
        return 1
    for m in re.finditer(r"\d+", text):
        v = int(m.group())
        if 1 <= v <= n:
            return v
    return 1

# Helper function to parse the confidence response from the LLM.
def _parse_confidence(text: str | None) -> int:
    if not text:
        return 5
    m = re.search(r"\b(10|[1-9])\b", text.strip())
    return int(m.group(1)) if m else 5

# Binary Judge is a function that judges if the prediction is correct or not. It returns True if the prediction is correct and False otherwise.
def binary_judge(question, prediction, expected, model, temperature=0.0) -> bool:
    system = "You are a strict grader. Reply with exactly True or False. No explanation."
    user = (
        f"Question: {question}\nPrediction: {prediction}\nExpected: {expected}\n"
        "Is the prediction correct? True or False"
    )
    r = _chat(system, user, model, temperature)
    if not r.get("ok"):
        return str(prediction).strip() == str(expected).strip()
    return _parse_bool(r.get("text"), prediction, expected)

# Comparative Judge is a function that judges if the prediction is correct or not. It returns the index of the best answer.
def comparative_judge(question, candidates, model, temperature=0.0) -> int:
    n = len(candidates)
    if n < 2:
        return 1
    n = min(n, 3)
    parts = candidates[:n]
    if n == 2:
        system = "You are a judge. Choose the best answer from the candidates. Reply with only the number (1 or 2)."
    else:
        system = "You are a judge. Choose the best answer from the candidates. Reply with only the number (1, 2, or 3)."
    lines = "\n".join(f"Candidate {i + 1}: {parts[i]}" for i in range(n))
    user = f"Question: {question}\n{lines}\nWhich candidate has the best answer?"
    r = _chat(system, user, model, temperature)
    return _parse_choice(r.get("text"), n)

# Confidence Check is a function that checks the confidence of the answer. It returns the confidence score.
def confidence_check(question, answer, model, temperature=0.0) -> int:
    system = (
        "Rate your confidence in this answer from 1 (very unsure) to 10 (certain). "
        "Reply with just the number."
    )
    user = f"Question: {question}\nAnswer: {answer}\nConfidence (1-10):"
    r = _chat(system, user, model, temperature)
    return _parse_confidence(r.get("text"))
