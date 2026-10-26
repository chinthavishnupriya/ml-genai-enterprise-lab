"""Independent scanner agents, remediation advisor, and QA reviewer."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
from .models import AgentStatus, Finding
from .scanners import ConfigurationScanner, DependencyScanner, SecretsScanner, StaticCodeScanner, discover_files

class Scanner(Protocol):
    name: str
    def scan(self, path: Path, content: str, root: Path) -> list[Finding]: ...

class RemediationAdvisor:
    """Adds human-reviewable guidance; it does not edit or execute repository code."""
    name = "remediation_advisor"

    @staticmethod
    def advise(finding: Finding) -> str:
        return finding.recommendation

class QAReviewer:
    """Checks severity, required evidence and duplicate fingerprints."""
    name = "qa_reviewer"

    def review(self, findings: list[Finding]) -> tuple[list[Finding], list[str]]:
        accepted: list[Finding] = []
        errors: list[str] = []
        seen = set()
        for finding in findings:
            if finding.severity not in {"critical", "high", "medium", "low", "info"}:
                errors.append(f"{finding.rule_id}: invalid severity")
                continue
            if not finding.evidence or not finding.recommendation:
                errors.append(f"{finding.rule_id}: missing evidence or recommendation")
                continue
            if finding.fingerprint in seen:
                continue
            seen.add(finding.fingerprint)
            accepted.append(finding)
        return accepted, errors

class ScannerAgent:
    def __init__(self, scanner: Scanner):
        self.scanner = scanner
        self.name = scanner.name

    def run(self, files: list[Path], root: Path) -> tuple[list[Finding], AgentStatus]:
        findings: list[Finding] = []
        errors: list[str] = []
        scanned = 0
        for path in files:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                scanned += 1
                findings.extend(self.scanner.scan(path, content, root))
            except OSError as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
        status = AgentStatus(
            agent=self.name,
            files_scanned=scanned,
            findings=len(findings),
            status="PASS" if not errors else "PARTIAL",
            errors=tuple(errors[:20]),
        )
        return findings, status

def default_agents() -> list[ScannerAgent]:
    return [
        ScannerAgent(SecretsScanner()),
        ScannerAgent(StaticCodeScanner()),
        ScannerAgent(ConfigurationScanner()),
        ScannerAgent(DependencyScanner()),
    ]

def collect_files(root: str | Path, exclude_paths=()) -> list[Path]:
    return discover_files(root, exclude_paths=exclude_paths)
