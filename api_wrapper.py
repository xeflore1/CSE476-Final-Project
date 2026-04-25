import os, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

def call_model_chat_completions(messages=None,
                     model: str = MODEL,
                     temperature: float = 0.3,
                     frequency_penalty: float = 0.0,
                     max_tokens: int = 2048,
                     max_retries: int = 1,
                     timeout: int = 120) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    print(f"""API running with params:\n
    messages: {messages}\n
    model: {model}\n
    temp: {temperature}\n
    frequency_penalty: {frequency_penalty}\n
    max tokens: {max_tokens}\n
    max_retries: {max_retries}\n
    timeout: {timeout}\n""")
          
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        # "messages": [
        #     {"role": "system", "content": system},
        #     {"role": "user",   "content": prompt}
        # ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "stop": ["<DONE>"]
    }
    bad_dict = {}
    for i in range(max_retries):
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
                bad_dict = {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
                continue
        except requests.RequestException as e:
            bad_dict = {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}
    return bad_dict; # if all retries failed, return the most recent error logs