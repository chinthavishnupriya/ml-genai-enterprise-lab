"""Evaluate the transparent rules baseline on the synthetic held-out scenario group."""
from __future__ import annotations
import json
from .baseline import route_claim
from .data import split_examples


def evaluate_baseline() -> dict:
    test = split_examples()["test"]
    route_ok = action_ok = missing_docs_ok = 0
    results = []
    for item in test:
        expected = item["target"]
        actual = route_claim(item["claim_type"], item["claim_id"], item["submitted_documents"], item["claim_text"])
        route_match = actual["recommended_queue"] == expected["recommended_queue"]
        action_match = actual["next_step"] == expected["next_step"]
        missing_match = actual["missing_documents"] == expected["missing_documents"]
        route_ok += int(route_match)
        action_ok += int(action_match)
        missing_docs_ok += int(missing_match)
        results.append({
            "claim_id": item["claim_id"],
            "route_match": route_match,
            "next_step_match": action_match,
            "missing_documents_match": missing_match,
            "human_review_required": actual["human_review_required"],
        })
    n = len(test)
    return {
        "evaluation_mode": "rules_baseline",
        "test_examples": n,
        "route_accuracy": round(route_ok / n, 4) if n else 0.0,
        "next_step_accuracy": round(action_ok / n, 4) if n else 0.0,
        "missing_documents_accuracy": round(missing_docs_ok / n, 4) if n else 0.0,
        "results": results,
        "note": "Generated labels share the documented routing rules; these scores validate program logic only, not real claims accuracy.",
    }


def main() -> int:
    print(json.dumps(evaluate_baseline(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
