# Project 08 — Collaborative Multi-Agent DevSecOps Audit

## 1. Project overview

Perform a read-only static audit of source code and configuration files. Deterministic scanner components produce structured findings, a remediation advisor attaches guidance, a QA stage validates/deduplicates findings, and an orchestrator emits reports plus an event trace. The scanners do not execute the audited application or automatically change its files.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | File discovery, scanner logic, finding models, orchestration and CLI |
| Python standard library | Regex patterns, JSON, file traversal and output writing |
| JSON / Markdown / JSONL | Machine-readable findings, human-readable report and event log |
| `unittest` | Scanner rules, discovery filters and dependency-check tests |

The default project has no runtime third-party dependencies and does not use an external vulnerability database.

## 3. Scanner agents

- **Secrets scanner:** identifies credential-like assignments, private-key headers, AWS access-key-shaped strings and credentials embedded in URLs; report evidence is redacted.
- **Static-code scanner:** checks selected patterns for dynamic execution, shell-enabled subprocess calls, unsafe deserialization, weak hashes, disabled TLS verification and suspicious SQL interpolation.
- **Configuration scanner:** checks patterns such as explicit root container users, broad workflow permissions, `pull_request_target` workflows and broad Terraform ingress.
- **Dependency scanner:** detects unconstrained Python requirements and floating JavaScript versions. Direct URL requirements are checked for a same-line pip hash. A hash is an integrity control, not a safety guarantee.
- **Remediation advisor:** attaches reviewable recommendations; it does not patch the target.
- **QA reviewer/orchestrator:** checks finding shape, deduplicates identical findings, counts severities and records the execution trace.

## 4. File discovery and boundaries

Discovery filters excluded directories (for example, virtual environments, node_modules, build outputs and audit reports), skips symlinks, restricts source/config extensions and ignores files over the configured size limit. The audit output directory is excluded to reduce the risk of recursively scanning its own output.

Rules are syntactic heuristics. A finding means that a pattern needs review, not that the code is exploitable.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/scanners.py` | Discovery filters and four main scanner classes |
| `src/models.py` | Structured finding and report data models |
| `src/agents.py` | Remediation/QA agent behavior |
| `src/audit.py` | Orchestration, summary and report writing |
| `src/cli.py` | Audit CLI and output destination |
| `fixtures/vulnerable_sample/` | Intentionally insecure, dummy-only scanner test fixture |
| `tests/test_scanners.py` | Scanner rules and file discovery |
| `tests/test_audit.py` | Orchestration/report tests |
| `VALIDATION_REPORT.json` | Recorded fixture audit evidence |

## 6. Run locally

```bash
python -m unittest discover -s tests -v
python -m src.cli --path fixtures/vulnerable_sample --output-dir /tmp/devsecops-audit --json
```

Do not execute the intentionally vulnerable fixture. It should be treated as static test input only.

## 7. Recorded verification

The earlier Colab audit scanned five fixture files and reported seven findings (four high, one medium, two low), with a heuristic risk score of 34/100. After adding direct-URL dependency regression coverage, GitHub Actions passed all 19 tests. The fixture results describe expected scanner behavior on that sample; they do not establish the security of other repositories.

## 8. Limitations and future improvements

- No live CVE database, package-advisory API, license scanner or supply-chain attestation is included.
- Regex scanners can miss complex or obfuscated cases and generate false positives.
- Add allowlisted rule configuration, SARIF export, richer dependency lockfile support and rule-versioning if expanding the tool.
- Validate findings with security engineering and run remediations in an isolated environment.
- The score is heuristic triage, not compliance certification.
