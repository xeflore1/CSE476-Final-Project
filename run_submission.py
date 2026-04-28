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


def _load_existing_answers() -> list[dict]:
    if not OUTPUT_FILE.exists():
        return []
    try:
        with OUTPUT_FILE.open(encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_checkpoint(answers: list[dict]) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, indent=2)


def generate_answers():
    with INPUT_FILE.open(encoding='utf-8') as fp:
        test_data = json.load(fp)
    answers = _load_existing_answers()
    start_idx = len(answers)
    if start_idx:
        print(f"Resuming from checkpoint: {start_idx}/{len(test_data)}")
    start = time.time()
    for i in range(start_idx, len(test_data)):
        row = test_data[i]
        q = row["input"]
        print(f"\n{'='*60}\n[Q {i+1}/{len(test_data)}] domain=pending  len={len(q)}\n{'='*60}")
        try:
            out = agent(q)
        except Exception as e:
            out = f"ERROR: {e}"
        answers.append({"input": q, "output": out if out is not None else ""})
        # Save progress incrementally so work survives interruptions/crashes.
        if (i + 1) % 10 == 0:
            _save_checkpoint(answers)
        if (i + 1) % 100 == 0:
            dt = time.time() - start
            n = len(test_data)
            print(
                f"  {i + 1}/{n} in {dt:.1f}s "
                f"(~{dt / (i + 1):.2f}s/q, eta {dt / (i + 1) * (n - i - 1) / 60:.1f} min)"
            )
    _save_checkpoint(answers)
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
    with OUTPUT_FILE.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, indent=2)
    print(f"Wrote {len(answers)} answers to {OUTPUT_FILE}")


def main():
    answers = generate_answers()
    validate_results(answers)
    save_answers(answers)


if __name__ == "__main__":
    main()
