import os, json, textwrap, re, time
import requests
from dotenv import load_dotenv; 
load_dotenv()
API_KEY = os.getenv("API_KEY")
print(repr(API_KEY))
from api_wrapper import call_model_chat_completions
from utils import extract_answer


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
    
def self_refine(prompt: str, domain: str = "", max_iterations: int = 1) -> dict:
    transcript= []
    i = 0
    answer1 = ask(prompt)["text"]
    for i in range(max_iterations):
        feedback = ask(f"Give your critique of this answer:\n\n {answer1}")["text"]
        answer2 = ask(f"""I will give you the original question, an answer, and a critique. Please give a better answer to the original question using the
                      insight from the critique.
                      Question: {prompt}
                      Original Answer: {answer1}
                      Critique: {feedback}""")["text"]
        
        transcript.append({"answer": answer1, "feedback": feedback, "improved answer": answer2})
        i += 1
        answer1 = answer2
    
    return {"final answer": answer1, "transcript": transcript}