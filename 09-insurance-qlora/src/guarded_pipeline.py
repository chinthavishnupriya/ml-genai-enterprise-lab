"""Hybrid claim pipeline: deterministic fields, model-generated summary as untrusted draft."""
from __future__ import annotations
import json
import re
from typing import Any
from .baseline import route_claim

FORBIDDEN_SUMMARY_TERMS = (
    "approve", "approved", "deny", "denied", "reject", "rejected",
    "fraud", "liability", "liable", "coverage", "covered", "payout", "pay the claimant",
)
MAX_SUMMARY_CHARS = 400


def _model_summary(model_output: Any) -> str | None:
    """Extract only a draft summary. Every other model-produced field is ignored."""
    payload = model_output
    if isinstance(model_output, str):
        start, end = model_output.find("{"), model_output.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(model_output[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, str):
        return None
    summary = " ".join(summary.split()).strip()
    if not summary or len(summary) > MAX_SUMMARY_CHARS:
        return None
    if any(term in summary.casefold() for term in FORBIDDEN_SUMMARY_TERMS):
        return None
    return summary


def guarded_triage(example: dict[str, Any], model_output: Any = None) -> dict[str, Any]:
    """Build authoritative routing fields using deterministic rules, never the model's fields.

    The input claim ID/type/submitted-document list must come from trusted, validated
    application data. Only a short, screened narrative summary may be taken from a
    model response; it is explicitly labeled as a draft for human verification.
    """
    if not isinstance(example, dict):
        raise ValueError("Example input must be an object")
    claim_id = example.get("claim_id")
    claim_type = example.get("claim_type")
    submitted = example.get("submitted_documents")
    description = example.get("claim_text", "Synthetic claim")
    if not isinstance(claim_id, str) or not re.fullmatch(r"SYN-[A-Za-z0-9_-]{1,32}", claim_id):
        raise ValueError("Trusted input must have a valid synthetic claim ID")
    if not isinstance(claim_type, str):
        raise ValueError("Trusted input must contain claim_type")
    if not isinstance(submitted, list) or not all(isinstance(doc, str) for doc in submitted):
        raise ValueError("Trusted input submitted_documents must be a list of strings")

    result = route_claim(claim_type, claim_id, submitted, str(description))
    candidate = _model_summary(model_output)
    if candidate:
        # Avoid misleading draft text that names a different synthetic record.
        mentioned_ids = set(re.findall(r"\bSYN-[A-Za-z0-9_-]+\b", candidate))
        if mentioned_ids and mentioned_ids != {claim_id}:
            candidate = None
    if candidate:
        result["summary"] = f"Model-generated draft for human verification: {candidate}"
        result["summary_source"] = "model_draft_requires_human_verification"
    else:
        result["summary_source"] = "deterministic_fallback"
    # Reassert this invariant at the final boundary regardless of model contents.
    result["human_review_required"] = True
    return result
