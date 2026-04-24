import os
from dotenv import load_dotenv
<<<<<<< HEAD
from api_wrapper import call_LLM
=======
from techniques.api_wrapper import call_LLM
>>>>>>> origin/main
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Chain of thought algorithm
def chain_of_thought(prompt: str,
                     system: str = 
                        "You are a logical assistant. Think step-by-step and explain your reasoning clearly before answering." 
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>",
                        temperature: float = 0.3
                     ) -> dict:

    print(f"COT is running with prompt: {prompt}\n")
    
    return call_LLM(prompt, system, temperature=temperature, max_tokens=8192, timeout=120)