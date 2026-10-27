# Project 08 — Collaborative Multi-Agent DevSecOps Audit

**Status:** Verified in Google Colab and GitHub Actions. The original 17 tests passed in Colab; after fixing direct-URL dependency detection and adding two regression tests, the revised 19-test suite passed in CI.

## Objective

Run a safe, read-only static audit of a repository and produce actionable, structured findings. The orchestration is deterministic and local: files are read as text and no repository source is executed, installed, automatically patched, or uploaded.

## Components

- **Secrets agent:** scans for credential-like assignments, private-key headers, AWS access-key-shaped strings, and credentials embedded in URLs. Potential secret values are not copied into finding evidence.
- **Static-code agent:** flags selected code patterns such as dynamic code execution, shell-enabled subprocess calls, unsafe deserialization, weak hash functions, disabled TLS verification, and interpolated SQL.
- **Configuration agent:** checks for selected risky configuration patterns, such as a root container user, broad GitHub Actions permissions, a risky pull-request trigger, or open Terraform ingress.
- **Dependency agent:** notes unpinned Python requirements, direct URL requirements without a same-line hash, and floating JavaScript dependency versions. It is not a vulnerability database and does not identify CVEs.
- **Remediation advisor:** attaches human-reviewable guidance; it does not modify files.
- **QA reviewer:** checks required finding fields and deduplicates identical findings.
- **Orchestrator:** records per-agent status, severity counts, a simple capped risk score, and timestamped audit events.

## Quick start

Python 3.10+ is recommended. The default implementation uses only the Python standard library.

~~~bash
python -m unittest discover -s tests -v
python -m src.cli --path fixtures/vulnerable_sample --output-dir /tmp/devsecops-audit --json
~~~

The sample fixture is intentionally insecure and contains dummy test values only. **Do not execute the fixture code.** It exists to demonstrate how the static scanners report suspicious patterns.

## Google Colab

For a fresh runtime:

~~~python
%cd /content
!git clone https://github.com/chinthavishnupriya/ml-genai-enterprise-lab.git
%cd /content/ml-genai-enterprise-lab/08-devsecops-multi-agent
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.cli --path fixtures/vulnerable_sample --output-dir /content/devsecops-audit --json
~~~

If the repository is already cloned in this runtime, run !git -C /content/ml-genai-enterprise-lab pull origin main and then change into the Project 8 directory.

## Recorded Colab verification

- Original Colab tests: 17 passed, 0 failed.
- Revised GitHub Actions tests: 19 passed, 0 failed (run [40463898783](https://github.com/chinthavishnupriya/ml-genai-enterprise-lab/actions/runs/40463898783)).
- Synthetic fixture audit: 5 files scanned; 7 findings (4 high, 1 medium, 2 low).
- Heuristic risk score: 34/100, classified as high.
- Each of the four scanner agents completed with PASS status. These values describe the intentionally insecure fixture only, not the security of an arbitrary application.

## Output artifacts

The audit command writes:

- audit_report.json — structured findings, fingerprints, severity summary, agent status, and audit events.
- audit_report.md — human-readable review report with recommendations.
- audit_log.jsonl — one JSON record per orchestrator event.

The default output directory is audit-reports/. This directory is excluded from discovery to prevent scanning the generated output recursively.

## Safety, interpretation and limitations

- The tool performs static text checks only. It does not execute scanned code, run package installers, modify source, open network connections, or auto-apply patches.
- Findings are pattern matches, not proof of exploitability. False positives and false negatives are expected.
- Severity and risk score are heuristic triage aids, not compliance scores or certifications.
- Dependency checks inspect version-specification hygiene only. Direct URL dependencies are checked for a same-line pip hash; hash presence does not establish package safety. No live advisory database or CVE feed is queried.
- The included vulnerable fixture is intentionally insecure for scanner testing; findings on that fixture are expected.
- Have a security engineer review findings, validate context, and test remediations in an isolated environment before merging changes into real systems.
