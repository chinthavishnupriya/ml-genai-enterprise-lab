"""Deterministic synthetic claims generation and leakage-aware scenario split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

from .schema import CLAIM_TYPES

REQUIRED_DOCUMENTS = {
    "auto_collision": ["incident_report", "damage_photos", "repair_estimate"],
    "property_damage": ["damage_photos", "repair_estimate", "ownership_proof"],
    "medical_expense": ["itemized_bill", "proof_of_payment", "provider_note"],
    "travel_disruption": ["travel_itinerary", "carrier_disruption_notice", "expense_receipts"],
}

SCENARIOS = [
    ("S01", "a low-speed vehicle collision in a car park"),
    ("S02", "storm-related damage to a residential roof"),
    ("S03", "an emergency visit followed by a medical expense claim"),
    ("S04", "a delayed flight and an overnight hotel expense"),
    ("S05", "a rear-end vehicle collision during rainy weather"),
    ("S06", "water damage discovered after a pipe leak"),
    ("S07", "a follow-up clinic visit with a submitted medical bill"),
    ("S08", "a cancelled connecting flight and extra transport costs"),
]


def build_examples() -> list[dict[str, Any]]:
    examples = []
    idx = 0
    for scenario_id, description in SCENARIOS:
        for claim_type in sorted(CLAIM_TYPES):
            required = REQUIRED_DOCUMENTS[claim_type]
            for complete in (True, False):
                idx += 1
                if complete:
                    submitted = required.copy()
                    missing: list[str] = []
                else:
                    missing = [required[(idx + len(scenario_id)) % len(required)]]
                    submitted = [doc for doc in required if doc not in missing]
                claim_id = f"SYN-{idx:04d}"
                next_step = "request_missing_documents" if missing else "queue_for_human_review"
                amount = 650 + (idx * 379) % 7400
                claim_text = (
                    f"Synthetic claim type: {claim_type}. Incident description: {description}. "
                    f"Estimated amount: USD {amount}. Submitted documents: {', '.join(submitted) or 'none'}. "
                    "No real person, policy, or insurer is represented."
                )
                target = {
                    "claim_id": claim_id,
                    "claim_type": claim_type,
                    "recommended_queue": claim_type,
                    "next_step": next_step,
                    "missing_documents": missing,
                    "human_review_required": True,
                    "summary": (
                        f"Synthetic {claim_type} claim. "
                        + (f"Request {', '.join(missing)} before the file is reviewed." if missing
                           else "Documents listed for the example are present; route to a qualified human reviewer.")
                    ),
                }
                examples.append({
                    "claim_id": claim_id,
                    "scenario_group": scenario_id,
                    "claim_type": claim_type,
                    "instruction": "Extract a safe claim-routing summary from this synthetic record. Do not determine coverage, liability, fraud, payout, or denial. Always require human review.",
                    "claim_text": claim_text,
                    "submitted_documents": submitted,
                    "required_documents": required,
                    "target": target,
                })
    return examples


def format_prompt(example: dict[str, Any]) -> str:
    return (
        "### Task\n"
        + example["instruction"]
        + "\n\n### Synthetic claim record\n"
        + example["claim_text"]
        + "\n\n### Required output\n"
        + "Return only a JSON object with keys: claim_id, claim_type, recommended_queue, next_step, "
        "missing_documents, human_review_required, summary. Set human_review_required to true. "
        "Only request missing documents or route to a human reviewer; never decide claim approval, denial, payment, liability, or fraud.\n\n"
        + "### Response JSON\n"
    )


def split_examples(examples: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Split by scenario group so closely templated variants cannot cross partitions."""
    examples = examples if examples is not None else build_examples()
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in examples:
        by_group.setdefault(item["scenario_group"], []).append(item)
    groups = sorted(by_group)
    train_groups, validation_groups, test_groups = set(groups[:6]), set(groups[6:7]), set(groups[7:])
    return {
        "train": [item for item in examples if item["scenario_group"] in train_groups],
        "validation": [item for item in examples if item["scenario_group"] in validation_groups],
        "test": [item for item in examples if item["scenario_group"] in test_groups],
    }


def write_splits(output_dir: str | Path = "data/processed") -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = split_examples()
    counts = {}
    for name, records in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = len(records)
    metadata = {
        "dataset": "synthetic insurance claims triage demo",
        "split_strategy": "scenario-group split",
        "counts": counts,
        "scenario_groups": {name: sorted({row["scenario_group"] for row in rows}) for name, rows in splits.items()},
        "warning": "Synthetic data only; not for coverage, liability, fraud, payout, or denial decisions.",
    }
    (output / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and split fictional synthetic claim instructions.")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    print(json.dumps(write_splits(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
