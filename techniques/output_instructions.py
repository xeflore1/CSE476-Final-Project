from __future__ import annotations

_OUTPUT = {
    "math": """You need to answer the question in the following format:
Double check your arithmetic and your answer is a number.
{
    "answer": "answer",
}""",
    "common_sense": """You need to answer the question in the following format:
{
    "answer": "answer",
}""",
    "coding": """You need to answer the question in the following format:
{
    "answer": "answer",
}""",
    "future_prediction": """You need to answer the question in the following format:
{
    "answer": "\\boxed{answer}",
}""",
    "planning": """You need to answer the question in the following format:
{
    "answer": "action1\naction2\naction3\n...",
}""",
}


def output_instructions(domain: str) -> str:
    return _OUTPUT.get(domain, _OUTPUT["common_sense"]).strip()
