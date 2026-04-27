import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524}


def _should_retry(status: int) -> bool:
    if status == -1:
        return True
    return status in _RETRY_STATUS


def call_model_chat_completions(payload=None, *,
                                messages=None,
                                model: str = MODEL,
                                temperature: float = 0.0,
                                frequency_penalty: float = 0.0,
                                max_tokens: int = 2048,
                                max_retries: int = 3,
                                timeout: int = 120) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None,
      'status': int, 'error': str or None, 'headers': dict, 'calls': int }

    Retries transient failures (429, 5xx, network errors) with exponential
    backoff. Does NOT retry 4xx errors that won't change on retry (e.g. 400
    context-length, 401 auth, 404 not found).
    """

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    if payload is None:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "stop": ["<DONE>"],
        }

    last_result: dict = {}
    total_calls = 0
    backoff = 1.0

    for attempt in range(max(1, max_retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            total_calls += 1
            status = resp.status_code
            hdrs = dict(resp.headers)

            if status == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if attempt > 0:
                    print(f"[api] 200 OK after retry #{attempt}")
                return {
                    "ok": True,
                    "text": text,
                    "raw": data,
                    "status": status,
                    "error": None,
                    "headers": hdrs,
                    "calls": total_calls,
                }

            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            err_snippet = str(err_body)[:200].replace("\n", " ")
            print(f"[api] non-200 status={status} attempt={attempt + 1}/{max_retries} err={err_snippet}")

            last_result = {
                "ok": False,
                "text": None,
                "raw": None,
                "status": status,
                "error": str(err_body),
                "headers": hdrs,
                "calls": total_calls,
            }

            if not _should_retry(status):
                return last_result

        except requests.RequestException as e:
            print(f"[api] network error attempt={attempt + 1}/{max_retries} err={str(e)[:200]}")
            last_result = {
                "ok": False,
                "text": None,
                "raw": None,
                "status": -1,
                "error": str(e),
                "headers": {},
                "calls": total_calls,
            }

        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    return last_result
