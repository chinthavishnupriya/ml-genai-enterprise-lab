"""Simple, explicit input/output safety filters for the offline demo."""
from __future__ import annotations
import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{7,}\d)(?!\w)")
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+#[^\]\n]{1,100})\]")
INJECTION_PATTERNS = (
    re.compile(r"\bignore (?:all )?(?:previous|prior|above) (?:instructions|prompts)\b", re.I),
    re.compile(r"\bdisregard (?:all )?(?:previous|prior|above) (?:instructions|prompts)\b", re.I),
    re.compile(r"\b(reveal|print|show|leak)\b.{0,30}\b(system prompt|hidden instructions|secrets|credentials)\b", re.I),
    re.compile(r"\b(bypass|disable|override)\b.{0,25}\b(safety|guardrails|security checks|policy)\b", re.I),
    re.compile(r"\b(exfiltrate|jailbreak|execute arbitrary code)\b", re.I),
)


@dataclass(frozen=True)
class InputCheck:
    allowed: bool
    sanitized_text: str
    blocked_reason: str | None
    redactions: int


def inspect_input(text: str, *, max_chars: int = 1200) -> InputCheck:
    if not isinstance(text, str):
        return InputCheck(False, "", "input_not_text", 0)
    stripped = text.strip()
    if not stripped:
        return InputCheck(False, "", "empty_question", 0)
    if len(stripped) > max_chars:
        return InputCheck(False, "", "question_too_long", 0)
    for pattern in INJECTION_PATTERNS:
        if pattern.search(stripped):
            return InputCheck(False, "", "possible_prompt_injection", 0)
    redactions = 0
    def replace_email(match):
        nonlocal redactions
        redactions += 1
        return "[REDACTED_EMAIL]"
    def replace_phone(match):
        nonlocal redactions
        redactions += 1
        return "[REDACTED_PHONE]"
    sanitized = EMAIL_RE.sub(replace_email, stripped)
    sanitized = PHONE_RE.sub(replace_phone, sanitized)
    return InputCheck(True, sanitized, None, redactions)


def validate_cited_output(answer: str, allowed_citations: set[str]) -> tuple[bool, str]:
    """Ensure every emitted citation was supplied by a trusted local source."""
    citations = CITATION_RE.findall(answer or "")
    if not citations:
        return False, "missing_citations"
    unknown = [citation for citation in citations if citation not in allowed_citations]
    if unknown:
        return False, "unknown_citation"
    return True, "ok"


def extract_citations(answer: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(CITATION_RE.findall(answer or "")))
