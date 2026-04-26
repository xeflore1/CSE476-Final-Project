import concurrent.futures
from collections import Counter
from utils import extract_answer
from techniques.chain_of_thought import chain_of_thought

# Self-consistency algorithm
def self_consistency(prompt: str,
                     domain: str = "common_sense",
                     system: str = (
                        "You are a logical assistant. Think step-by-step and answer the given question."
                        "Your final answer MUST end with this exact format:\n"
                        "\\boxed{answer}\n"
                        "<DONE>"
                     ),
                     temperature: float = 0.5, # increase temp so that model explores different logical approaches
                     max_tokens: int = 4000,
                     timeout: int = 120,
                     n_samples: int = 4,
                     **_ignored
                    ):

    # Run chain of thought 4 times and pick the most frequent answer
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_samples) as executor:
        futures = {
            executor.submit(
                chain_of_thought, prompt, domain,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            ) for i in range(n_samples)
        }
        results = []
        calls_total = 0
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            results.append(response)
            calls_total += response.get("calls", 0)
        responses = []
        for response in results:
            if response.get("ok"):
                answer = response.get("answer") or extract_answer(response.get("text", ""))
                responses.append(answer)
        print(f"Responses: {responses}")
        filteredList = [x for x in responses if x is not None and str(x).strip() != ""] # get rid of None/empty entries
        print(f"filered list: {filteredList}")
        if not filteredList:
            return {"ok": False, "text": None, "answer": "", "calls": calls_total, "error": "no valid samples"}
        else:
            counts = Counter(filteredList)
            most_common_val = counts.most_common(1)[0][0] # find the most common answer
            return {"ok": True, "text": most_common_val, "answer": most_common_val, "calls": calls_total, "error": None} # return the most common answer
                
