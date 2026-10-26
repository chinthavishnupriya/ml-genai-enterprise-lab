import json
import tempfile
import unittest
from pathlib import Path

from src.scanners import (
    ConfigurationScanner, DependencyScanner, SecretsScanner, StaticCodeScanner,
    discover_files,
)


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.code = self.root / "app.py"

    def tearDown(self):
        self.temp.cleanup()

    def test_secret_assignment_is_detected_without_value(self):
        self.code.write_text('API_KEY = "static-test-value-456789"\n', encoding="utf-8")
        findings = SecretsScanner().scan(self.code, self.code.read_text(), self.root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].rule_id, "SEC001")
        self.assertNotIn("static-test-value-456789", findings[0].evidence)

    def test_placeholder_secret_is_ignored(self):
        self.code.write_text('API_KEY = "replace-me-with-real-key"\n', encoding="utf-8")
        findings = SecretsScanner().scan(self.code, self.code.read_text(), self.root)
        self.assertEqual(findings, [])

    def test_private_key_header_is_detected(self):
        findings = SecretsScanner().scan(self.code, "-----BEGIN PRIVATE KEY-----\n", self.root)
        self.assertTrue(any(f.rule_id == "SEC002" for f in findings))

    def test_eval_is_flagged(self):
        findings = StaticCodeScanner().scan(self.code, "answer = eval(user_text)\n", self.root)
        self.assertTrue(any(f.rule_id == "PY001" for f in findings))

    def test_shell_true_is_flagged(self):
        findings = StaticCodeScanner().scan(self.code, "subprocess.run(command, shell=True)\n", self.root)
        self.assertTrue(any(f.rule_id == "PY002" for f in findings))

    def test_safe_yaml_loader_not_flagged(self):
        findings = StaticCodeScanner().scan(self.code, "yaml.safe_load(source)\n", self.root)
        self.assertFalse(any(f.rule_id == "PY003" for f in findings))

    def test_disabled_tls_is_flagged(self):
        findings = StaticCodeScanner().scan(self.code, "requests.get(url, verify=False)\n", self.root)
        self.assertTrue(any(f.rule_id == "PY005" for f in findings))

    def test_root_container_user_is_flagged(self):
        docker = self.root / "Dockerfile"
        findings = ConfigurationScanner().scan(docker, "FROM python:3.12\nUSER root\n", self.root)
        self.assertTrue(any(f.rule_id == "CFG001" for f in findings))

    def test_workflow_write_all_is_flagged(self):
        workflow = self.root / ".github" / "workflows" / "ci.yml"
        workflow.parent.mkdir(parents=True)
        findings = ConfigurationScanner().scan(workflow, "permissions: write-all\n", self.root)
        self.assertTrue(any(f.rule_id == "CFG002" for f in findings))

    def test_unpinned_requirement_is_flagged(self):
        req = self.root / "requirements.txt"
        findings = DependencyScanner().scan(req, "requests\n", self.root)
        self.assertTrue(any(f.rule_id == "DEP001" for f in findings))

    def test_pinned_requirement_is_not_flagged_as_unpinned(self):
        req = self.root / "requirements.txt"
        findings = DependencyScanner().scan(req, "requests==2.32.4\n", self.root)
        self.assertFalse(any(f.rule_id == "DEP001" for f in findings))

    def test_floating_package_json_dependency_is_flagged(self):
        package = self.root / "package.json"
        findings = DependencyScanner().scan(package, json.dumps({"dependencies": {"demo-lib": "latest"}}), self.root)
        self.assertTrue(any(f.rule_id == "DEP003" for f in findings))

    def test_discovery_skips_virtualenv_and_large_files(self):
        (self.root / "app.py").write_text("print('safe')\n", encoding="utf-8")
        (self.root / "large.py").write_bytes(b"x" * 1_000_010)
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "hidden.py").write_text("eval(x)", encoding="utf-8")
        files = discover_files(self.root)
        self.assertEqual([p.name for p in files], ["app.py"])


if __name__ == "__main__":
    unittest.main()
