import os, json, textwrap, re, time
import requests
from dotenv import load_dotenv; 
load_dotenv()
API_KEY = os.getenv("API_KEY")
print(repr(API_KEY))


API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")         

def ask(prompt, system: str ="You are a helpful assistant ready to answer a question.", model: str = MODEL, temperature: float = 0.0, timeout: int = 60) -> dict:
    """
    Basic ask function
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    } 

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            print("200 returned")
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            print("200 is NOT returned")
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

def tree_of_thought(prompt: str) -> dict:
    thought1 = ask(prompt)
    thought2 = ask("Come up with a different approach to this one for the question:\n\n" + (thought1["text"] or ""))
    thought3 = ask("Come up with an approach not like the two provided for the question:\n\n" + (thought2["text"] or "") 
                   + (thought1["text"] or ""))
    finalanswer = ask("Choose one of these three final approaches to use as the basis for answering your question. Try to make" \
    "your best judgement regarding which of the approaches is best. Remember to give a final answer at the end:\n\n" + (thought1["text"] or "") 
    + "\n\n" + (thought2["text"] or "") + "\n\n" + (thought3["text"] or ""))
    return {"thought1": thought1, "thought2": thought2, "thought3": thought3, "finalanswer": finalanswer}