from __future__ import annotations

from api_wrapper import MODEL, call_model_chat_completions


def prompt_optimized_call(prompt, domain, model, temperature, max_tokens, timeout) -> dict:
    print(f"Prompt optimization is running with prompt: {prompt}\n")

    def math_guidlines():
        return (
            "You rewrite math word-problems into a clearer form. "
            "Do NOT solve the problem. "
            "Return ONLY the rewritten problem statement on a single line, "
            "preserving every numeric value, symbol, and constraint exactly."
        )

    def common_sense_guidlines():
        return (
            "You rewrite common-sense and reading questions for clarity. "
            "Do NOT answer the question. "
            "Return ONLY the rewritten question on a single line, "
            "preserving all entities, options, and factual constraints exactly."
        )

    def coding_guidlines():
        return (
            "You rewrite coding tasks to be clearer and more explicit. "
            "Do NOT write or solve code. "
            "Return ONLY the rewritten coding task on a single line, "
            "preserving function signatures, required imports, I/O expectations, and constraints exactly."
        )

    def future_prediction_guidlines():
        return (
            "You rewrite future-prediction questions to be precise and unambiguous. "
            "Do NOT make a prediction. "
            "Return ONLY the rewritten prediction question on a single line, "
            "preserving every date, option, formatting requirement, and constraint exactly."
        )

    def planning_guidlines():
        return (
            "You rewrite planning problems into a clearer task specification. "
            "Do NOT produce an action plan. "
            "Return ONLY the rewritten planning problem on a single line, "
            "preserving all initial states, goals, allowed actions, and constraints exactly."
        )

    domain_prompts = {
        "math": math_guidlines,
        "common_sense": common_sense_guidlines,
        "coding": coding_guidlines,
        "future_prediction": future_prediction_guidlines,
        "planning": planning_guidlines,
    }
    guide_fn = domain_prompts.get(domain, common_sense_guidlines)
    system_content = guide_fn().strip()

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    try:
        res = call_model_chat_completions(
            messages=messages,
            temperature=temperature,
            frequency_penalty=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            model=model or MODEL,
        )
        if not res["ok"]:
            return {"optimized": prompt, "calls": res.get("calls", 1)}
        return {"optimized": res["text"], "calls": res.get("calls", 1)}
    except Exception as e:
        return {"optimized": prompt, "calls": 0, "error": str(e)}