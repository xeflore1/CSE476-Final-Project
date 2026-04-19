from __future__ import annotations

from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
import finalProject as final_project
from techniques.output_instructions import output_instructions

JUDGE_MAX_TOKENS = 32

# Binary Judge is a function that judges if the prediction is correct or not. It returns True if the prediction is correct and False otherwise.
def binary_judge(question, prediction, expected, model, temperature=0.0) -> bool:
    # Calls the LLM endpoint with the binary judge prompt. It is similar to the self_evaluate function in the final_project_tutorial.ipynb.
    prompt = f"""
    You are a strict grader. Reply with exactly True or False. No explanation.
    Question: {question}
    Prediction: {prediction}
    Expected: {expected}
    Is the prediction correct? True or False
    """
    return final_project.call_model_chat_completions(prompt, model=model, temperature=temperature, max_tokens=JUDGE_MAX_TOKENS)

# Comparative Judge is a function that judges if the prediction is correct or not. It returns the index of the best answer.
def comparative_judge(question, candidates, model, temperature=0.0) -> int:
    # Calls the LLM endpoint with the comparative judge prompt. It is similar to the self_evaluate function in the final_project_tutorial.ipynb.
    prompt = f"""
    You are a judge. Choose the best answer from the candidates. Reply with only the number (1, 2, or 3).
    Question: {question}
    Candidate 1: {candidates[0]}
    Candidate 2: {candidates[1]}
    Candidate 3: {candidates[2]}
    Which candidate has the best answer?
    """
    return final_project.call_model_chat_completions(prompt, model=model, temperature=temperature, max_tokens=JUDGE_MAX_TOKENS)

# Confidence Check is a function that checks the confidence of the answer. It returns the confidence score.
def confidence_check(question, answer, model, temperature=0.0) -> int:
    # Calls the LLM endpoint with the confidence check prompt. It is similar to the self_evaluate function in the final_project_tutorial.ipynb.
    prompt = f"""
    You are a confidence checker. Rate your confidence in this answer from 1 (very unsure) to 10 (certain). Reply with just the number.
    Question: {question}
    Answer: {answer}
    Confidence (1-10):
    """
    return final_project.call_model_chat_completions(prompt, model=model, temperature=temperature, max_tokens=JUDGE_MAX_TOKENS)
