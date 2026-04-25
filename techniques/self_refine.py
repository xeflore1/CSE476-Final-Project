from api_wrapper import call_model_chat_completions, MODEL
from utils import extract_answer

def self_refine(prompt: str, domain: str = "", system: str = "You are a helpful assistant ready to answer a question.", model: str = MODEL, temperature: float = 0.0,
                 timeout: int = 60, max_iterations: int = 1,) -> dict:
    transcript = []
    def ask(user_prompt: str):
        try:
            response = call_model_chat_completions(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
                timeout=timeout,
                model=model,
            )
            return {"ok": True, "text": response.get("text"), "answer": response.get("text"), "calls": response.get("calls", 0), "error": None}
        except Exception as e:
            return {"ok": False, "text": None, "answer": None, "calls": 0, "error": str(e)}
        
    prompt1 = prompt
    answer1 = ask(prompt1)
    for i in range(max_iterations):
        feedback = ask(f"Critique this answer:\n\n{answer1['text'] or ''}")
        prompt1 = f"{prompt}\n\nPrevious answer:\n{answer1['text'] or ''}\n\nFeedback:\n{feedback['text'] or ''}\n\nPlease provide an improved answer."
        answer2 = ask(prompt1)
        transcript.append({
            "iteration": i + 1,
            "answer": answer2,
            "feedback": feedback,
        })
        answer1 = answer2
    text = (answer1 or {}).get("text") or ""
    calls_total = sum(t.get("answer", {}).get("calls", 0) + t.get("feedback", {}).get("calls", 0) for t in transcript)
    calls_total += (answer1 or {}).get("calls", 0)  # initial ask
    return {
        "ok": bool((answer1 or {}).get("ok")),
        "text": text,
        "answer": extract_answer(text) or text.strip(),
        "calls": calls_total,
        "error": (answer1 or {}).get("error"),
    }