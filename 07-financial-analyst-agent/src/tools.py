"""Controlled offline tools: read-only SQLite, safe calculator, and local document search."""
from __future__ import annotations
import ast
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SEED_ROWS = [
    ("ACME Systems", "2025-Q1", 1200000, 168000, 510000, 240000),
    ("ACME Systems", "2025-Q2", 1260000, 176400, 545000, 235000),
    ("ACME Systems", "2025-Q3", 1320000, 184800, 570000, 228000),
    ("ACME Systems", "2025-Q4", 1400000, 210000, 615000, 220000),
    ("Beacon Retail", "2025-Q1", 980000, 78400, 320000, 180000),
    ("Beacon Retail", "2025-Q2", 1020000, 81600, 342000, 176000),
    ("Beacon Retail", "2025-Q3", 1060000, 74200, 355000, 190000),
    ("Beacon Retail", "2025-Q4", 1100000, 88000, 380000, 188000),
    ("Cobalt Manufacturing", "2025-Q1", 1550000, 155000, 410000, 360000),
    ("Cobalt Manufacturing", "2025-Q2", 1500000, 135000, 392000, 380000),
    ("Cobalt Manufacturing", "2025-Q3", 1620000, 178200, 420000, 365000),
    ("Cobalt Manufacturing", "2025-Q4", 1700000, 204000, 465000, 350000),
]
ALLOWED_TABLES = {"quarterly_financials"}
ALLOWED_FUNCTIONS = {"count", "sum", "avg", "min", "max", "round", "coalesce", "lower", "upper", "abs", "total"}


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    summary: str
    data: Any = None
    source: str | None = None


def initialize_demo_database(path: str | Path) -> Path:
    """Create synthetic financial data only when the database does not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE TABLE quarterly_financials (
                company TEXT NOT NULL, quarter TEXT NOT NULL,
                revenue_usd INTEGER NOT NULL CHECK (revenue_usd >= 0),
                operating_income_usd INTEGER NOT NULL,
                cash_usd INTEGER NOT NULL CHECK (cash_usd >= 0),
                debt_usd INTEGER NOT NULL CHECK (debt_usd >= 0),
                PRIMARY KEY (company, quarter)
            )"""
        )
        connection.executemany("INSERT INTO quarterly_financials VALUES (?, ?, ?, ?, ?, ?)", SEED_ROWS)
        connection.commit()
    finally:
        connection.close()
    return path


def _readonly_authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    """Allow SELECT/read/function actions on the one approved table."""
    if action in {sqlite3.SQLITE_SELECT, getattr(sqlite3, "SQLITE_RECURSIVE", -1)}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or arg1 or "").casefold()
        return sqlite3.SQLITE_OK if function_name in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


class ReadOnlySQLTool:
    """Read-only SQL with bound parameters, an allowlisted table, and row/VM limits."""
    def __init__(self, database_path: str | Path, *, max_rows: int = 100) -> None:
        self.database_path = initialize_demo_database(database_path)
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.max_rows = min(max_rows, 500)

    def query(self, sql: str, params: Sequence[Any] = ()) -> ToolObservation:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL query must be a non-empty string")
        if len(sql) > 1500:
            raise ValueError("SQL query exceeds the maximum length")
        stripped = sql.strip()
        if not re.match(r"(?is)^SELECT\b", stripped):
            raise ValueError("Only SELECT statements are allowed")
        if ";" in stripped or "--" in stripped or "/*" in stripped or "*/" in stripped:
            raise ValueError("Multi-statement SQL and SQL comments are not allowed")
        uri = self.database_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=3.0)
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(_readonly_authorizer)
        steps = [0]
        def limit_vm_steps() -> int:
            steps[0] += 1
            return 1 if steps[0] > 10000 else 0
        connection.set_progress_handler(limit_vm_steps, 1000)
        try:
            cursor = connection.execute(stripped, tuple(params))
            columns = [desc[0] for desc in cursor.description or []]
            rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            rows = rows[:self.max_rows]
            data = [dict(row) for row in rows]
            return ToolObservation(
                "read_only_sql",
                f"Read {len(data)} row(s) from the approved synthetic financial table" + ("; result truncated" if truncated else ""),
                {"columns": columns, "rows": data, "truncated": truncated},
                "SQLite synthetic demo database",
            )
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Read-only SQL query rejected: {exc}") from exc
        finally:
            connection.close()


_ALLOWED_AST = (
    ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.Pow, ast.Mod, ast.UAdd, ast.USub,
)


class SafeCalculator:
    """Evaluate bounded arithmetic expressions without eval or arbitrary code."""
    def calculate(self, expression: str) -> ToolObservation:
        if not expression or len(expression) > 200:
            raise ValueError("Expression must be non-empty and at most 200 characters")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid arithmetic expression") from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST):
                raise ValueError(f"Unsupported calculator operation: {type(node).__name__}")
            if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
                raise ValueError("Only numeric literals are allowed")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, (int, float)) or abs(node.right.value) > 8:
                    raise ValueError("Exponent must be a numeric constant between -8 and 8")
        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant):
                value = float(node.value)
                if abs(value) > 1e15:
                    raise ValueError("Numeric literal is too large")
                return value
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): result = left + right
                elif isinstance(node.op, ast.Sub): result = left - right
                elif isinstance(node.op, ast.Mult): result = left * right
                elif isinstance(node.op, ast.Div):
                    if right == 0: raise ValueError("Division by zero")
                    result = left / right
                elif isinstance(node.op, ast.Mod):
                    if right == 0: raise ValueError("Modulo by zero")
                    result = left % right
                elif isinstance(node.op, ast.Pow): result = left ** right
                else: raise ValueError("Unsupported operator")
                if not math.isfinite(result) or abs(result) > 1e18:
                    raise ValueError("Calculated result is outside the supported range")
                return result
            raise ValueError("Unsupported expression")
        value = evaluate(tree)
        return ToolObservation("calculator", f"Calculated {expression} = {value:.8g}", {"expression": expression, "value": value})


class LocalDocumentSearchTool:
    """Simple section-based search over local Markdown/text research notes."""
    def __init__(self, document_dir: str | Path) -> None:
        self.document_dir = Path(document_dir)

    def search(self, query: str, *, top_k: int = 3) -> ToolObservation:
        if not query.strip():
            raise ValueError("Document search query must not be blank")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        # Remove question filler so content and section titles drive relevance.
        search_stopwords = {
            "what", "are", "the", "main", "and", "for", "how", "when", "where",
            "is", "was", "were", "with", "from", "about", "should", "can", "does",
            "do", "this", "that", "their", "they", "you", "your", "please", "show",
            "tell", "me", "give", "some", "any", "to", "of", "in", "on", "a", "an",
        }
        raw_terms = {term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 2}
        terms = raw_terms - search_stopwords
        query_terms = terms | {term[:-1] for term in terms if len(term) > 4 and term.endswith("s")}
        results = []
        for path in sorted(self.document_dir.rglob("*")) if self.document_dir.exists() else []:
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            heading = path.stem
            body = []
            chunks = []
            for line in text.splitlines():
                match = re.match(r"^\s{0,3}#{1,6}\s+(.+)", line)
                if match:
                    if body:
                        chunks.append((heading, "\n".join(body).strip()))
                    heading = match.group(1).strip()
                    body = []
                else:
                    body.append(line)
            if body:
                chunks.append((heading, "\n".join(body).strip()))
            for section, chunk in chunks:
                chunk_terms_raw = set(re.findall(r"[a-z0-9]+", (section + " " + chunk).casefold()))
                chunk_terms = chunk_terms_raw | {
                    term[:-1] for term in chunk_terms_raw if len(term) > 4 and term.endswith("s")
                }
                overlap = query_terms & chunk_terms
                if overlap:
                    title_terms_raw = set(re.findall(r"[a-z0-9]+", section.casefold()))
                    title_terms = title_terms_raw | {
                        term[:-1] for term in title_terms_raw if len(term) > 4 and term.endswith("s")
                    }
                    title_overlap = len(query_terms & title_terms)
                    score = len(overlap) / max(1, len(query_terms)) + 0.25 * title_overlap
                    results.append({"source": path.name, "section": section, "text": chunk, "score": round(score, 4)})
        results.sort(key=lambda x: (-x["score"], x["source"], x["section"]))
        results = results[:top_k]
        return ToolObservation("document_search", f"Retrieved {len(results)} research passage(s)", results, "local synthetic research notes")
