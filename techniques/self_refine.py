from api_wrapper import call_model_chat_completions, MODEL
from utils import extract_answer

def self_refine(prompt, system: str ="You are a helpful assistant ready to answer a question.", model: str = MODEL, temperature: float = 0.0, timeout: int = 60) -> dict:
    """
    Basic ask function | UPDATED TO USE API WRAPPER
    """
    try:
        resp = call_model_chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=2048,
            timeout=timeout,
        )
        if resp.get("ok"):
            print("200 returned")
            return {
                "ok": True,
                "text": resp.get("text"),
                "raw": resp.get("raw"),
                "status": resp.get("status"),
                "error": None,
                "headers": resp.get("headers", {}),
                "calls": resp.get("calls", 0),
            }
        else:
            print("200 is NOT returned")
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "status": resp.get("status", -1),
                "error": str(resp.get("error")),
                "headers": resp.get("headers", {}),
                "calls": resp.get("calls", 0),
            }
    except Exception as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}, "calls": 0}
    
def self_refinement(prompt: str) -> dict:
    answer1 = self_refine(prompt)
    feedback1 = self_refine("Give your critique of this answer:\n\n" + (answer1["text"] or ""))
    answer2 = self_refine("using this critique, give a better answer to the original question:\n\n" + feedback1["text"] or "")
    return {"answer1": answer1, "feedback1": feedback1, "answer2": answer2}