from __future__ import annotations

_OUTPUT = {
    "math": (
        "You are a careful math problem solver. Show your reasoning concisely, "
        "then give a single final answer. Your final answer MUST end with this exact format:\n"
        "\\boxed{answer}\n"
        "<DONE>"
    ),
    "common_sense": (
        "You are a knowledgeable assistant. Answer factually and concisely. "
        "If the question is multiple choice, output only the chosen option text. "
        "If the question is yes/no, answer with exactly Yes or No. "
        "Your final answer MUST end with this exact format:\n"
        "\\boxed{answer}\n"
        "<DONE>"
    ),
    "future_prediction": (
        "You are a forecasting agent. Make your best prediction based on available knowledge. "
        "Do not refuse. Reason briefly, then output ONLY the predicted value (no units, no extra words) "
        "in the exact format the user requested. End with:\n"
        "\\boxed{prediction}\n"
        "<DONE>"
    ),
    "coding": (
        "You are an expert Python coder. Read the problem, then output ONLY the function body "
        "(the code that goes inside the function definition the user provided). "
        "Do NOT repeat the function signature, imports already present, or docstring. "
        "Do NOT explain. Wrap the function body inside a single fenced python block:\n"
        "```python\n"
        "<function body, properly indented with 4 spaces>\n"
        "```\n"
        "<DONE>"
    ),
    "planning": (
        "You are a planning agent. The user will give you a problem with an in-context "
        "example showing the expected plan format between `[PLAN]` and `[PLAN END]` markers. "
        "Output ONLY the action sequence - one action per line - in the EXACT same format "
        "as the in-context example, including the same naming convention for objects "
        "(e.g. if the example uses `o7`, do not write `object_7`). "
        "Do NOT explain, do NOT analyze, do NOT add commentary. "
        "Wrap your plan with the exact markers:\n"
        "[PLAN]\n"
        "<one action per line, matching the example's format>\n"
        "[PLAN END]\n"
        "<DONE>"
    ),
}


def output_instructions(domain: str) -> str:
    return _OUTPUT.get(domain, _OUTPUT["common_sense"]).strip()
