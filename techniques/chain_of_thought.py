import os
from api_wrapper import call_model_chat_completions
from utils import extract_answer

# Chain of thought algorithm  prompt, model, temperature, timeout, max_tokens
def chain_of_thought(prompt: str,
                     domain: str = "common_sense",
                     system: str = 
                        "You are a logical assistant. Think step-by-step and explain your reasoning clearly before answering." 
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>",
                     temperature: float = 0.3,
                     timeout: int = 120,
                     max_tokens: int = 8000,
                     **_ignored
                     ) -> dict:

    print(f"COT is running with prompt: {prompt}\n")
    _ = domain
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt}
    ]
    res = call_model_chat_completions(messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    text = res.get("text")
    return {
        "ok": bool(res.get("ok")),
        "text": text,
        "answer": extract_answer(text) or (text or "").strip(),
        "calls": res.get("calls", 0),
        "error": res.get("error")
    }