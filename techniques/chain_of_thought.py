import os
from dotenv import load_dotenv
from api_wrapper import call_model_chat_completions
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Chain of thought algorithm  prompt, model, temperature, timeout, max_tokens
def chain_of_thought(prompt: str,
                     system: str = 
                        "You are a logical assistant. Think step-by-step and explain your reasoning clearly before answering." 
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>",
                     temperature: float = 0.3,
                     timeout: int = 120,
                     max_tokens: int = 8000,
                     ) -> dict:

    print(f"COT is running with prompt: {prompt}\n")
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt}
    ]
    return call_model_chat_completions(messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)