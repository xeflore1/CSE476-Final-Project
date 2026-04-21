import os, json, textwrap, re, time
import requests
import concurrent.futures
from collections import Counter
from dotenv import load_dotenv
from utils import extract_answer
from technique.chain_of_thought import chain_of_thought
load_dotenv()

API_KEY  = os.getenv('API-KEY')
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Self-consistency algorithm
def self_consistency(prompt: str,
                     system: str = (
                        "You are a logical assistant. Think step-by-step and answer the given question."
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>"
                    ),
                     model: str = MODEL,
                     temperature: float = 0.5, # increase temp so that model explores different logical approaches
                     timeout: int = 120):

    # Run chain of thought 4 times and pick the most frequent answer
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(chain_of_thought, prompt, system, model, temperature, timeout) for i in range(4)}
        responses = []
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            if response["ok"]:
                answer = extract_answer(response["text"])
                responses.append(answer)
        print(f"Responses: {responses}")
        filteredList = [x for x in responses if x is not None] # get rid of None entries
        print(f"filered list: {filteredList}")
        if not filteredList:
            return None
        else:
            counts = Counter(filteredList)
            most_common_val = counts.most_common(1)[0][0] # find the most common answer
            return most_common_val # return the most common answer
                
