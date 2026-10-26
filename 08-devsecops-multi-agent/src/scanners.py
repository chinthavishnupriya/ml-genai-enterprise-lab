"""Read-only deterministic static checks for defensive audit patterns."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .models import Finding

MAX_FILE_BYTES = 1_000_000
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", "site-packages", "audit-reports",
}
SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".tf", ".sh", ".ps1", ".sql"}
SPECIAL_NAMES = {"Dockerfile", "dockerfile", "requirements.txt", "Pipfile", "package.json", ".env", ".env.example"}
PLACEHOLDER_VALUES = {
    "", "changeme", "change_me", "replace_me", "replace-me", "your_api_key",
    "your_api_key_here", "example", "placeholder", "redacted", "dummy", "fake",
    "not-a-real-secret", "not_real", "todo", "none", "null",
}
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)\b(api[_-]?key|access[_-]?key|client[_-]?secret|secret[_-]?key|password|passwd|token|auth[_-]?token)\b\s*[:=]\s*(['"])([^'"\r\n]{8,})\2"""
)
PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
URL_CREDENTIAL = re.compile(r"https?://[^/\s:@]+:[^/\s@]+@")

PYTHON_RULES = [
    ("PY001", "Dynamic code execution", "high", re.compile(r"\b(eval|exec)\s*\("),
     "Avoid evaluating untrusted strings; replace with explicit parsing or a constrained interpreter."),
    ("PY002", "Subprocess with shell enabled", "high", re.compile(r"\bshell\s*=\s*True\b"),
     "Pass an argument list to subprocess and keep shell=False; validate arguments and use timeouts."),
    ("PY003", "Unsafe deserialization", "high", re.compile(r"\bpickle\.loads?\s*\(|\byaml\.load\s*\("),
     "Use safe, schema-validated formats; for YAML use safe_load/SafeLoader and avoid untrusted pickle data."),
    ("PY004", "Weak hash algorithm", "medium", re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("),
     "Use a modern hash such as SHA-256 for integrity; use a dedicated password-hashing scheme for passwords."),
    ("PY005", "TLS certificate verification disabled", "high", re.compile(r"\bverify\s*=\s*False\b"),
     "Keep TLS certificate verification enabled; configure a trusted CA bundle when necessary."),
    ("PY006", "Potential SQL built by string interpolation", "high",
     re.compile(r"""(?:execute|executemany)\s*\(\s*(?:f["']|["'][^"']*\{\s*\w+\s*\})"""),
     "Use parameterized SQL placeholders and pass values separately; never concatenate untrusted values into SQL."),
]
CONFIG_RULES = [
    ("CFG001", "Container explicitly runs as root", "medium", re.compile(r"(?im)^\s*USER\s+root\s*(?:#.*)?$"),
     "Create and select a non-root application user in the container image."),
    ("CFG002", "Broad GitHub Actions workflow permissions", "high", re.compile(r"(?im)^\s*permissions\s*:\s*write-all\s*$"),
     "Set least-privilege workflow permissions explicitly at the workflow/job level."),
    ("CFG003", "Potentially risky pull_request_target workflow", "high", re.compile(r"(?is)\bon:\s*.*?pull_request_target\b"),
     "Review untrusted pull-request code paths carefully; avoid checking out/executing contributor code with write-capable tokens."),
    ("CFG004", "Terraform resource may expose broad ingress", "medium", re.compile(r'(?is)cidr_blocks\s*=\s*\[\s*"0\.0\.0\.0/0"\s*\]'),
     "Restrict ingress to known required networks and document the exposure."),
]


def _is_placeholder(value: str) -> bool:
    stripped = value.strip().strip("'\" ").casefold()
    if stripped in PLACEHOLDER_VALUES:
        return True
    return any(marker in stripped for marker in ("example_placeholder", "redacted", "replace-me", "your-secret", "fake-secret", "not-a-real"))


def discover_files(root: str | Path, *, exclude_paths: Iterable[str | Path] = ()) -> list[Path]:
    root = Path(root).resolve()
    excluded = {Path(p).resolve() for p in exclude_paths}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if any(path.resolve() == item or item in path.resolve().parents for item in excluded):
            continue
        if path.name not in SPECIAL_NAMES and path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.append(path)
    return sorted(files)


def _finding(rule_id: str, title: str, severity: str, category: str, path: Path, root: Path,
             line: int | None, description: str, evidence: str, recommendation: str, agent: str) -> Finding:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return Finding(rule_id, title, severity, category, relative, line, description, evidence, recommendation, agent)


class SecretsScanner:
    name = "secrets"
    def scan(self, path: Path, content: str, root: Path) -> list[Finding]:
        findings = []
        for number, line in enumerate(content.splitlines(), start=1):
            patterns = [
                ("SEC001", "Hardcoded credential-like assignment", SECRET_ASSIGNMENT),
                ("SEC002", "Private key material in source", PRIVATE_KEY_HEADER),
                ("SEC003", "AWS access-key-shaped string", AWS_ACCESS_KEY),
                ("SEC004", "Credentials embedded in URL", URL_CREDENTIAL),
            ]
            for rule_id, title, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                if rule_id == "SEC001" and _is_placeholder(match.group(3)):
                    continue
                findings.append(_finding(
                    rule_id, title, "critical" if rule_id in {"SEC002", "SEC003"} else "high",
                    "secrets", path, root, number,
                    "Potential secret material detected; confirm whether the value is active.",
                    "Sensitive value redacted from report.",
                    "Remove the value from source control, rotate/revoke it if it was real, and load it from an approved secret manager or runtime environment.",
                    self.name,
                ))
                break
        return findings


class StaticCodeScanner:
    name = "static_code"
    def scan(self, path: Path, content: str, root: Path) -> list[Finding]:
        findings = []
        if path.suffix.lower() not in {".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".ps1", ".sql"}:
            return findings
        for number, line in enumerate(content.splitlines(), start=1):
            for rule_id, title, severity, pattern, recommendation in PYTHON_RULES:
                if pattern.search(line):
                    category = "injection" if rule_id == "PY006" else "code_safety"
                    findings.append(_finding(
                        rule_id, title, severity, category, path, root, number,
                        "Static pattern suggests a security-sensitive construct; inspect context before classifying it as exploitable.",
                        f"Matched rule {rule_id}; source line omitted from report.",
                        recommendation, self.name
                    ))
        return findings


class ConfigurationScanner:
    name = "configuration"
    def scan(self, path: Path, content: str, root: Path) -> list[Finding]:
        findings = []
        special = path.name in {"Dockerfile", "dockerfile"} or ".github/workflows" in path.as_posix() or path.suffix.lower() == ".tf"
        if not special:
            return findings
        for number, line in enumerate(content.splitlines(), start=1):
            for rule_id, title, severity, pattern, recommendation in CONFIG_RULES:
                if pattern.search(line):
                    findings.append(_finding(
                        rule_id, title, severity, "configuration", path, root, number,
                        "Configuration pattern may broaden privileges or network exposure; review surrounding context.",
                        f"Matched rule {rule_id}; surrounding configuration omitted.",
                        recommendation, self.name
                    ))
        return findings


class DependencyScanner:
    name = "dependencies"
    def scan(self, path: Path, content: str, root: Path) -> list[Finding]:
        findings = []
        if path.name == "requirements.txt":
            for number, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line or line.startswith(("#", "-", "git+", "http://", "https://")):
                    continue
                package = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
                if package and not re.search(r"(?:===|==|~=|<=|>=|<|>)", line):
                    findings.append(_finding(
                        "DEP001", "Unpinned Python dependency", "low", "dependencies", path, root, number,
                        f"Dependency '{package}' has no version constraint; installs may not be reproducible.",
                        f"Package name: {package}; no version constraint detected.",
                        "Set and periodically update a tested version range or lock dependency versions with a reproducible lockfile.",
                        self.name
                    ))
                if re.search(r"(?:https?|git)\+?://", line) and "--hash=" not in line:
                    findings.append(_finding(
                        "DEP002", "Direct URL dependency without hash", "medium", "dependencies", path, root, number,
                        "Direct URL dependency is not accompanied by a pip hash on this line.",
                        "Direct dependency source is not pinned by hash.",
                        "Prefer a trusted package index and pinned version; verify origin and integrity. Avoid installing from untrusted URLs.",
                        self.name
                    ))
        elif path.name == "package.json":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return findings
            deps = {}
            deps.update(payload.get("dependencies", {}) if isinstance(payload.get("dependencies", {}), dict) else {})
            deps.update(payload.get("devDependencies", {}) if isinstance(payload.get("devDependencies", {}), dict) else {})
            for package, version in deps.items():
                if isinstance(version, str) and version.strip() in {"*", "latest"}:
                    findings.append(_finding(
                        "DEP003", "Floating JavaScript dependency version", "low", "dependencies", path, root, None,
                        f"Dependency '{package}' uses a floating version specifier.",
                        f"Package name: {package}; version specifier: {version}.",
                        "Pin a tested version or use the project's lockfile and a reproducible installation command.",
                        self.name
                    ))
        return findings
