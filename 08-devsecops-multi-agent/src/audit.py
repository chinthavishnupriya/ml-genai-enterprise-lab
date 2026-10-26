"""Orchestrate scanner agents and write JSON/Markdown audit artifacts."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from .agents import QAReviewer, RemediationAdvisor, collect_files, default_agents
from .models import AuditReport, Finding
import json

SEVERITY_WEIGHTS = {"critical": 10, "high": 7, "medium": 4, "low": 1, "info": 0}

def _risk_score(findings: list[Finding]) -> tuple[int, str]:
    score = min(100, sum(SEVERITY_WEIGHTS[f.severity] for f in findings))
    if any(f.severity == "critical" for f in findings):
        level = "critical"
    elif score >= 25 or any(f.severity == "high" for f in findings):
        level = "high"
    elif score >= 8:
        level = "medium"
    elif score > 0:
        level = "low"
    else:
        level = "none"
    return score, level

def _markdown(report: AuditReport) -> str:
    tick = chr(96)
    lines = [
        "# DevSecOps Multi-Agent Audit Report", "",
        "- **Root:** " + tick + report.root + tick,
        f"- **Files scanned:** {report.files_scanned}",
        f"- **Risk score:** {report.risk_score}/100 ({report.risk_level})",
        "", "## Summary", "", "| Severity | Count |", "|---|---:|",
    ]
    for sev in ("critical", "high", "medium", "low", "info"):
        lines.append(f"| {sev.title()} | {report.summary.get(sev, 0)} |")
    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("No findings matched the enabled static rules.")
    for i, finding in enumerate(report.findings, 1):
        location = finding.file_path + (f":{finding.line}" if finding.line else "")
        lines.extend([
            f"### {i}. [{finding.severity.upper()}] {finding.title}", "",
            "- **Rule:** " + tick + finding.rule_id + tick,
            "- **Location:** " + tick + location + tick,
            f"- **Category / agent:** {finding.category} / {tick}{finding.agent}{tick}",
            f"- **Description:** {finding.description}",
            f"- **Evidence:** {finding.evidence}",
            f"- **Recommended action:** {RemediationAdvisor.advise(finding)}",
            "- **Fingerprint:** " + tick + finding.fingerprint + tick, "",
        ])
    lines.extend(["## Agent status", "", "| Agent | Files scanned | Findings | Status |", "|---|---:|---:|---|"])
    for agent in report.agents:
        lines.append(f"| {agent.agent} | {agent.files_scanned} | {agent.findings} | {agent.status} |")
    lines.extend([
        "", "## Limitations", "",
        "This is a rule-based static audit, not a full SAST/SCA platform or proof of exploitability. Findings can be false positives and must be reviewed by a security engineer. The scanner reads files as text and does not execute, install, or modify repository code.", ""
    ])
    return "\n".join(lines)

def run_audit(root: str | Path, output_dir: str | Path | None = None) -> AuditReport:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output_path = Path(output_dir).resolve() if output_dir else None
    files = collect_files(root, exclude_paths=([output_path] if output_path else ()))
    events = [{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "audit_started",
        "files_discovered": len(files),
        "execution_mode": "read_only_static_analysis",
    }]
    findings: list[Finding] = []
    statuses = []
    for agent in default_agents():
        agent_findings, status = agent.run(files, root)
        findings.extend(agent_findings)
        statuses.append(status)
        events.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "agent_completed",
            "agent": status.agent,
            "status": status.status,
            "files_scanned": status.files_scanned,
            "findings": status.findings,
        })
    findings, qa_errors = QAReviewer().review(findings)
    findings.sort(key=lambda f: (-SEVERITY_WEIGHTS[f.severity], f.file_path, f.line or 0, f.rule_id))
    summary = {sev: sum(1 for f in findings if f.severity == sev) for sev in ("critical", "high", "medium", "low", "info")}
    score, level = _risk_score(findings)
    events.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "qa_review_completed",
        "accepted_findings": len(findings),
        "rejected_or_flagged": len(qa_errors),
    })
    report = AuditReport(
        root=str(root), files_scanned=len(files), findings=tuple(findings),
        agents=tuple(statuses), risk_score=score, risk_level=level,
        summary=summary, audit_events=tuple(events),
    )
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "audit_report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        (output_path / "audit_report.md").write_text(_markdown(report), encoding="utf-8")
        (output_path / "audit_log.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return report
