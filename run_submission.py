#!/usr/bin/env python3
"""Run the full agent router on official test data and write the grader submission JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent_router import agent

_ROOT = Path(__file__).resolve().parent
INPUT_FILE = _ROOT / "cse_476_final_project_test_data.json"
OUTPUT_FILE = _ROOT / "cse_476_final_project_answers.json"
EXPECTED_COUNT = 6208


def generate_answers():
    with INPUT_FILE.open(encoding='utf-8') as fp:
        test_data = json.load(fp)
    answers = []
    start = time.time()
    for i, row in enumerate(test_data):
        q = row["input"]
        try:
            out = agent(q)
        except Exception as e:
            out = f"ERROR: {e}"
        answers.append({"input": q, "output": out if out is not None else ""})
        if (i + 1) % 100 == 0:
            dt = time.time() - start
            n = len(test_data)
            print(
                f"  {i + 1}/{n} in {dt:.1f}s "
                f"(~{dt / (i + 1):.2f}s/q, eta {dt / (i + 1) * (n - i - 1) / 60:.1f} min)"
            )
    return answers


def validate_results(answers):
    if len(answers) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} answers, got {len(answers)}")
    for idx, a in enumerate(answers):
        if not isinstance(a.get("output"), str):
            raise ValueError(f"Answer at index {idx} is not a string")
        if len(a["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(a['output'])} chars)."
            )


def save_answers(answers):
    with OUTPUT_FILE.open("w") as fp:
        json.dump(answers, fp, indent=2)
    print(f"Wrote {len(answers)} answers to {OUTPUT_FILE}")


def main():
    answers = generate_answers()
    validate_results(answers)
    save_answers(answers)


if __name__ == "__main__":
    main()
