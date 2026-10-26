import json
import tempfile
import unittest
from pathlib import Path

from src.audit import run_audit, _risk_score
from src.agents import QAReviewer
from src.models import Finding


class AuditTests(unittest.TestCase):
    def test_qa_reviewer_deduplicates_findings(self):
        finding = Finding("PY001", "Dynamic execution", "high", "code_safety", "app.py", 1,
                          "Potential issue", "Rule matched", "Avoid eval", "static_code")
        result, errors = QAReviewer().review([finding, finding])
        self.assertEqual(len(result), 1)
        self.assertEqual(errors, [])

    def test_risk_score_and_level(self):
        finding = Finding("SEC002", "Private key", "critical", "secrets", "app.py", 1,
                          "Potential key", "Header matched", "Rotate key", "secrets")
        score, level = _risk_score([finding])
        self.assertEqual(score, 10)
        self.assertEqual(level, "critical")

    def test_audit_writes_report_and_log_without_executing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "target"
            out = Path(tmp) / "results"
            root.mkdir()
            (root / "bad.py").write_text('API_KEY = "static-test-value-456789"\nanswer = eval(user_input)\n', encoding="utf-8")
            report = run_audit(root, out)
            self.assertEqual(report.files_scanned, 1)
            self.assertTrue(any(f.rule_id == "SEC001" for f in report.findings))
            self.assertTrue(any(f.rule_id == "PY001" for f in report.findings))
            self.assertTrue((out / "audit_report.json").is_file())
            self.assertTrue((out / "audit_report.md").is_file())
            self.assertTrue((out / "audit_log.jsonl").is_file())
            payload = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["files_scanned"], 1)
            self.assertNotIn("static-test-value-456789", (out / "audit_report.json").read_text(encoding="utf-8"))

    def test_missing_root_fails_clearly(self):
        with self.assertRaises(NotADirectoryError):
            run_audit("/definitely/not/a/real/project/path")


if __name__ == "__main__":
    unittest.main()
