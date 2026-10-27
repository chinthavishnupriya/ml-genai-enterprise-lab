"""Evaluate deterministic guard behavior separately from raw adapter performance."""
from __future__ import annotations
import json
from .data import split_examples
from .guarded_pipeline import guarded_triage
from .schema import validate_triage_output


def evaluate_guarded_pipeline() -> dict:
    test = split_examples()["test"]
    checked = {
        "claim_id": 0, "claim_type": 0, "recommended_queue": 0, "next_step": 0,
        "missing_documents": 0, "human_review_required": 0, "valid_schema": 0,
    }
    rows = []
    for example in test:
        expected = example["target"]
        # Adversarial toy model response emulates the field-copy errors observed in
        # the raw adapter. The guard must not trust any structured model field.
        wrong_output = {
            "claim_id": "SYN-9999",
            "claim_type": "medical_expense" if example["claim_type"] != "medical_expense" else "auto_collision",
            "recommended_queue": "travel_disruption",
            "next_step": "queue_for_human_review" if expected["next_step"] == "request_missing_documents" else "request_missing_documents",
            "missing_documents": ["incorrect_document_name"],
            "human_review_required": False,
            "summary": "Synthetic draft; a qualified human should review the record.",
        }
        actual = guarded_triage(example, wrong_output)
        valid = True
        try:
            validate_triage_output(actual)
        except ValueError:
            valid = False
        checks = {
            "claim_id": actual["claim_id"] == expected["claim_id"],
            "claim_type": actual["claim_type"] == expected["claim_type"],
            "recommended_queue": actual["recommended_queue"] == expected["recommended_queue"],
            "next_step": actual["next_step"] == expected["next_step"],
            "missing_documents": actual["missing_documents"] == expected["missing_documents"],
            "human_review_required": actual["human_review_required"] is True,
            "valid_schema": valid,
        }
        for key, passed in checks.items():
            checked[key] += int(passed)
        rows.append({"claim_id": example["claim_id"], "checks": checks})
    n = len(test)
    return {
        "evaluation_mode": "deterministic_guard_unit_evaluation",
        "test_examples": n,
        "metrics": {key: round(value / n, 4) if n else 0.0 for key, value in checked.items()},
        "rows": rows,
        "note": "This deliberately supplies incorrect model fields to verify the deterministic guard. 100% metrics indicate guard behavior on synthetic examples, not language-model accuracy or real claims performance.",
    }


def main() -> int:
    print(json.dumps(evaluate_guarded_pipeline(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
