"""Validate the constrained output schema used in this educational project."""
from __future__ import annotations
from typing import Any

CLAIM_TYPES = {"auto_collision", "property_damage", "medical_expense", "travel_disruption"}
NEXT_STEPS = {"request_missing_documents", "queue_for_human_review"}
REQUIRED_OUTPUT_KEYS = {
    "claim_id", "claim_type", "recommended_queue", "next_step",
    "missing_documents", "human_review_required", "summary",
}


def validate_triage_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Triage output must be a JSON object")
    missing = REQUIRED_OUTPUT_KEYS - set(value)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if not isinstance(value["claim_id"], str) or not value["claim_id"].startswith("SYN-"):
        raise ValueError("claim_id must be a synthetic SYN-* identifier")
    if value["claim_type"] not in CLAIM_TYPES:
        raise ValueError("Unsupported claim_type")
    if value["recommended_queue"] != value["claim_type"]:
        raise ValueError("recommended_queue must match the supported claim_type")
    if value["next_step"] not in NEXT_STEPS:
        raise ValueError("Unsupported next_step")
    if not isinstance(value["missing_documents"], list) or not all(isinstance(x, str) for x in value["missing_documents"]):
        raise ValueError("missing_documents must be a list of strings")
    if not isinstance(value["human_review_required"], bool) or value["human_review_required"] is not True:
        raise ValueError("human_review_required must be true")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("summary must be a non-empty string")
    # Explicitly disallow model outputs that suggest an automatic payout or rejection.
    forbidden = ("approve claim", "deny claim", "reject claim", "pay the claimant", "fraudulent claimant")
    combined = " ".join([value["summary"], value["next_step"]]).casefold()
    if any(phrase in combined for phrase in forbidden):
        raise ValueError("Output contains an unsupported automated claim decision")
    return value
