"""Command-line entry point for the defensive audit prototype."""
import argparse
import json
import sys
from .audit import run_audit

def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only static DevSecOps checks. No source files are executed or modified.")
    parser.add_argument("--path", default=".", help="Repository root to scan")
    parser.add_argument("--output-dir", default="audit-reports", help="Output directory for JSON, Markdown and JSONL reports")
    parser.add_argument("--json", action="store_true", help="Print complete JSON report to stdout")
    parser.add_argument("--fail-on", choices=("none", "low", "medium", "high", "critical"), default="high")
    args = parser.parse_args()
    try:
        report = run_audit(args.path, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"DevSecOps audit complete: {report.files_scanned} files scanned; {len(report.findings)} finding(s).")
        print(f"Risk score: {report.risk_score}/100 ({report.risk_level})")
        print(f"Summary: {report.summary}")
        print(f"Reports written to: {args.output_dir}")
    fail_levels = {"none": -1, "low": 1, "medium": 2, "high": 3, "critical": 4}
    actual = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[report.risk_level]
    threshold = fail_levels[args.fail_on]
    return 1 if threshold >= 0 and actual >= threshold else 0

if __name__ == "__main__":
    raise SystemExit(main())
