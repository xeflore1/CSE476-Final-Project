from api_wrapper import call_model_chat_completions
from utils import extract_answer


def ask(prompt, system: str = "You are a helpful assistant ready to answer a question.", temperature: float = 0.0, max_tokens: int = 2048, timeout: int = 60) -> dict:
    """
    Basic ask function | UPDATED TO USE API WRAPPER
    """
    try:
        response = call_model_chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return {
            "ok": bool(response.get("ok")),
            "text": response.get("text"),
            "answer": response.get("text"),
            "calls": response.get("calls", 0),
            "error": response.get("error"),
        }
    except Exception as e:
        return {"ok": False, "text": None, "answer": None, "calls": 0, "error": str(e)}

def tree_of_thought(prompt: str, domain: str = "common_sense", *, max_tokens: int = 2048, timeout: int = 60, temperature: float = 0.0, **_ignored) -> dict:
    thought1 = ask(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    thought2 = ask("Come up with a different approach to this one for the question:\n\n" + (thought1["text"] or ""), temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    thought3 = ask("Come up with an approach not like the two provided for the question:\n\n" + (thought2["text"] or "")
                   + (thought1["text"] or ""), temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    finalanswer = ask("Choose one of these three final approaches to use as the basis for answering your question. Try to make"
    "your best judgement regarding which of the approaches is best. Remember to give a final answer at the end:\n\n" + (thought1["text"] or "")
    + "\n\n" + (thought2["text"] or "") + "\n\n" + (thought3["text"] or ""), temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return {
    "ok":     bool(finalanswer.get("ok")),
    "text":   finalanswer.get("text"),
    "answer": extract_answer(finalanswer.get("text")) or (finalanswer.get("text") or "").strip(),
    "calls":  thought1.get("calls", 0) + thought2.get("calls", 0) + thought3.get("calls", 0) + finalanswer.get("calls", 0),
    "error":  finalanswer.get("error"),
}