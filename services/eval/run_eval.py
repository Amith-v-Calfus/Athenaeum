"""Evaluation harness for the retrieval pipeline.

LIMITATION: this is a plain substring/source-match check, not a semantic
grader (no DeepEval/RAGAS). It tells you whether the answer contains the
expected facts and cites the expected file -- it does not judge fluency,
completeness, or whether extra/wrong claims were added. Good enough to
catch regressions quickly; not a substitute for a mentor's live read.

run_eval only works when user_id="eval_user"

Run against real, already-ingested data:
    python3 run_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from retrieval import answer_question  # noqa: E402

_EVAL_USER_ID = "eval_user"
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def check_answer(answer: str, expected_substrings: list[str], match: str) -> bool:
    answer_lower = answer.lower()
    hits = [s for s in expected_substrings if s.lower() in answer_lower]
    if match == "any":
        return len(hits) > 0
    return len(hits) == len(expected_substrings)


def check_sources(sources: list[dict], expected_source_file: str | None) -> bool:
    if expected_source_file is None:
        return True
    return any(s.get("filename") == expected_source_file for s in sources)


def main() -> None:
    with open(_DATASET_PATH) as f:
        dataset = json.load(f)

    passed = 0
    for i, case in enumerate(dataset, start=1):
        question = case["question"]
        expected_substrings = case["expected_answer_contains"]
        match = case.get("match", "all")
        expected_source_file = case.get("expected_source_file")

        result = answer_question(question, _EVAL_USER_ID)
        answer_ok = check_answer(result["answer"], expected_substrings, match)
        source_ok = check_sources(result["sources"], expected_source_file)
        ok = answer_ok and source_ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {i}. {question}")
        if not ok:
            print(f"    answer:  {result['answer']!r}")
            print(f"    sources: {result['sources']!r}")
            if not answer_ok:
                print(f"    expected ({match}) one of: {expected_substrings!r}")
            if not source_ok:
                print(f"    expected source: {expected_source_file!r}")

        if ok:
            passed += 1

    total = len(dataset)
    print()
    print(f"Passed {passed}/{total} ({passed / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
