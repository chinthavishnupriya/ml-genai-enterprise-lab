"""Shared structures for scanner findings and audit reports."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

SEVERITIES = {"critical", "high", "medium", "low", "info"}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    title: str
    severity: str
    category: str
    file_path: str
    line: int | None
    description: str
    evidence: str
    recommendation: str
    agent: str
    confidence: str = "high"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"Invalid severity: {self.severity}")
        if not self.rule_id or not self.title or not self.file_path:
            raise ValueError("Finding requires rule_id, title, and file_path")
        if self.line is not None and self.line < 1:
            raise ValueError("Line number must be positive")

    @property
    def fingerprint(self) -> str:
        raw = f"{self.rule_id}|{self.file_path}|{self.line or 0}|{self.title}"
        return sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True)
class AgentStatus:
    agent: str
    files_scanned: int
    findings: int
    status: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    root: str
    files_scanned: int
    findings: tuple[Finding, ...]
    agents: tuple[AgentStatus, ...]
    risk_score: int
    risk_level: str
    summary: dict[str, int]
    audit_events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
            "agents": [agent.to_dict() for agent in self.agents],
            "audit_events": list(self.audit_events),
            "disclaimer": "Static rule-based demonstration; findings require human review and are not a certification.",
        }
