"""Rules-only baseline used for quick offline smoke tests."""
from __future__ import annotations
from typing import Any
from .data import REQUIRED_DOCUMENTS
from .schema import validate_triage_output


def route_claim(claim_type: str, claim_id: str, submitted_documents: list[str], description: str = "Synthetic claim") -> dict[str, Any]:
    if claim_type not in REQUIRED_DOCUMENTS:
        raise ValueError(f"Unsupported claim type: {claim_type}")
    required = REQUIRED_DOCUMENTS[claim_type]
    submitted = set(submitted_documents)
    missing = [doc for doc in required if doc not in submitted]
    result = {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "recommended_queue": claim_type,
        "next_step": "request_missing_documents" if missing else "queue_for_human_review",
        "missing_documents": missing,
        "human_review_required": True,
        "summary": (
            f"Synthetic {claim_type} claim. "
            + (f"Request {', '.join(missing)} before the file is reviewed." if missing
               else "Documents listed for the example are present; route to a qualified human reviewer.")
        ),
    }
    return validate_triage_output(result)
