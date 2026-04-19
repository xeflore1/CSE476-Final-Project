from domain_classifier import classify_domain
from answer_extraction import extract_answer
from techniques.chain_of_thought import chain_of_thought
from techniques.self_consistency import self_consistency
from techniques.tree_of_thought import tree_of_thought
from techniques.self_refine import self_refine
from techniques.react_agent import react
from techniques.tool_augmented import tool_augmented
from techniques.decomposition import decomposition
from techniques.ensemble_voting import ensemble_vote
from techniques.prompt_optimization import prompt_optimized_call
from techniques.llm_as_judge import confidence_check


# Routing function is a function that routes the question to the appropriate technique.
def routing_function(question, domain):
    techniques = []
    if domain == "math":
        techniques.append(chain_of_thought)
        techniques.append(self_consistency)
    elif domain == "common_sense":
        techniques.append(prompt_optimized_call)
    elif domain == "coding":
        techniques.append(react)
        techniques.append(tool_augmented)
    elif domain == "future_prediction":
        techniques.append(chain_of_thought)
        techniques.append(self_consistency)
    elif domain == "planning":
        techniques.append(decomposition)
    return techniques

# Agent function is a function that routes the question to the appropriate technique and returns the answer.
def agent(question) -> str:
    domain = classify_domain(question)
    techniques = routing_function(question, domain)
    for technique in techniques:
        answer = technique(question)
        if answer is not None:
            return answer
    return "No answer found"

def track_llm_calls(techniques):
    for technique in techniques:
        if technique.__name__ == "chain_of_thought":
            return 1
        elif technique.__name__ == "self_consistency":
            return 1
        elif technique.__name__ == "tree_of_thought":
            return 1
        elif technique.__name__ == "self_refine":
            return 1
        elif technique.__name__ == "react":
            return 1

