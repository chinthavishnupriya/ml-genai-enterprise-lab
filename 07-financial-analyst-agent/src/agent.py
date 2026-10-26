"""Deterministic offline financial analyst agent using controlled tools."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .tools import LocalDocumentSearchTool, ReadOnlySQLTool, SafeCalculator, ToolObservation

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "financials_demo.sqlite"
DEFAULT_DOCS = Path(__file__).resolve().parents[1] / "data" / "research"


@dataclass(frozen=True)
class TraceEvent:
    step: int
    action: str
    tool_input: dict[str, Any]
    observation: str
    source: str | None = None


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    answer: str
    citations: tuple[str, ...]
    trace: tuple[TraceEvent, ...]
    abstained: bool


class FinancialAnalystAgent:
    """Pattern-based planner using read-only SQL, local document search and calculator."""
    def __init__(self, database_path: str | Path = DEFAULT_DB, document_dir: str | Path = DEFAULT_DOCS) -> None:
        self.sql = ReadOnlySQLTool(database_path)
        self.search = LocalDocumentSearchTool(document_dir)
        self.calculator = SafeCalculator()

    def _run_sql(self, trace: list[TraceEvent], sql: str, params: tuple[Any, ...], label: str) -> ToolObservation:
        obs = self.sql.query(sql, params)
        trace.append(TraceEvent(len(trace) + 1, "read_only_sql", {"label": label, "sql_template": sql, "params": list(params)}, obs.summary, obs.source))
        return obs

    def _run_calc(self, trace: list[TraceEvent], expression: str) -> float:
        obs = self.calculator.calculate(expression)
        trace.append(TraceEvent(len(trace) + 1, "calculator", {"expression": expression}, obs.summary, None))
        return float(obs.data["value"])

    def _run_search(self, trace: list[TraceEvent], query: str) -> ToolObservation:
        obs = self.search.search(query, top_k=3)
        trace.append(TraceEvent(len(trace) + 1, "document_search", {"query": query, "top_k": 3}, obs.summary, obs.source))
        return obs

    @staticmethod
    def _company_in_question(question: str) -> str | None:
        for name in ("ACME Systems", "Beacon Retail", "Cobalt Manufacturing"):
            if name.casefold() in question.casefold():
                return name
        return None

    @staticmethod
    def _quarter_in_question(question: str) -> str:
        match = re.search(r"\bq([1-4])\b", question.casefold())
        return f"2025-Q{match.group(1)}" if match else "2025-Q4"

    @staticmethod
    def _format_usd(value: Any) -> str:
        return chr(36) + f"{float(value):,.0f}"

    def answer(self, question: str) -> AgentAnswer:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be blank")
        trace: list[TraceEvent] = []
        q = question.strip()
        lowered = q.casefold()

        advice_terms = ("should i buy", "should i sell", "what stock should", "recommend a stock", "should i invest", "guaranteed return", "price target")
        if any(term in lowered for term in advice_terms):
            return AgentAnswer(
                q,
                "I cannot provide personalized investment recommendations or price targets. This demo works only with fictional financial records and local research notes; it has no current market data. I can summarize the sample financial metrics or retrieve the fictional risk notes instead.",
                (), (), True,
            )

        company = self._company_in_question(q)
        doc_terms = ("risk", "outlook", "assumption", "methodology", "macro", "liquidity", "leverage", "data quality", "model risk")
        if any(term in lowered for term in doc_terms) and not ("margin" in lowered and company):
            obs = self._run_search(trace, q)
            if not obs.data:
                return AgentAnswer(q, "No sufficiently relevant passage was found in the local fictional research notes. This demo cannot provide financial or investment advice.", (), tuple(trace), True)
            bullets, citations = [], []
            for row in obs.data[:2]:
                bullets.append(f"- **{row['section']}**: {row['text']}")
                citations.append(f"{row['source']} — {row['section']}")
            answer = "Relevant passages from fictional research notes (not financial advice):\n" + "\n".join(bullets)
            return AgentAnswer(q, answer, tuple(citations), tuple(trace), False)

        quarter = self._quarter_in_question(q)

        if "margin" in lowered and company:
            sql = "SELECT company, quarter, revenue_usd, operating_income_usd FROM quarterly_financials WHERE company = ? AND quarter = ?"
            obs = self._run_sql(trace, sql, (company, quarter), "Company operating margin inputs")
            rows = obs.data["rows"]
            if not rows:
                return AgentAnswer(q, f"No financial row was found for {company} in {quarter}.", (), tuple(trace), True)
            row = rows[0]
            expression = f"{row['operating_income_usd']} / {row['revenue_usd']} * 100"
            margin = self._run_calc(trace, expression)
            citation = f"quarterly_financials — {company}, {quarter}"
            answer = (
                f"**{company} operating margin, {quarter}: {margin:.2f}%**\n\n"
                f"Calculation: operating income ({self._format_usd(row['operating_income_usd'])}) / "
                f"revenue ({self._format_usd(row['revenue_usd'])}) × 100 = {margin:.2f}%.\n\n"
                "Source: synthetic demo database; these figures are fictional, not investment advice."
            )
            return AgentAnswer(q, answer, (citation,), tuple(trace), False)

        if "growth" in lowered or ("compare" in lowered and "revenue" in lowered):
            sql = "SELECT company, quarter, revenue_usd FROM quarterly_financials WHERE quarter IN (?, ?) ORDER BY company, quarter"
            obs = self._run_sql(trace, sql, ("2025-Q3", "2025-Q4"), "Q3-to-Q4 revenue comparison")
            by_company: dict[str, dict[str, int]] = {}
            for row in obs.data["rows"]:
                by_company.setdefault(row["company"], {})[row["quarter"]] = int(row["revenue_usd"])
            lines, citations = [], []
            for name, values in sorted(by_company.items()):
                if "2025-Q3" not in values or "2025-Q4" not in values:
                    continue
                growth = self._run_calc(trace, f"({values['2025-Q4']} - {values['2025-Q3']}) / {values['2025-Q3']} * 100")
                lines.append(f"- **{name}**: Q3 {self._format_usd(values['2025-Q3'])} → Q4 {self._format_usd(values['2025-Q4'])}; change {growth:+.2f}%.")
                citations.append(f"quarterly_financials — {name}, 2025-Q3 and 2025-Q4")
            return AgentAnswer(q, "Synthetic Q3-to-Q4 revenue comparison (fictional data, not a forecast):\n" + "\n".join(lines), tuple(citations), tuple(trace), False)

        if "cash" in lowered or "debt" in lowered:
            sql = "SELECT company, quarter, cash_usd, debt_usd FROM quarterly_financials WHERE quarter = ? ORDER BY company"
            obs = self._run_sql(trace, sql, (quarter,), "Cash and debt balances")
            lines = [
                f"- **{row['company']} ({row['quarter']})**: cash {self._format_usd(row['cash_usd'])}; debt {self._format_usd(row['debt_usd'])}."
                for row in obs.data["rows"]
            ]
            citations = tuple(f"quarterly_financials — {row['company']}, {row['quarter']}" for row in obs.data["rows"])
            return AgentAnswer(q, f"Cash and debt from the fictional dataset for {quarter}:\n" + "\n".join(lines), citations, tuple(trace), not bool(lines))

        sql = "SELECT company, quarter, revenue_usd, operating_income_usd, cash_usd, debt_usd FROM quarterly_financials WHERE quarter = ? ORDER BY revenue_usd DESC"
        obs = self._run_sql(trace, sql, (quarter,), "Latest-quarter financial summary")
        lines, citations = [], []
        for row in obs.data["rows"]:
            lines.append(
                f"- **{row['company']} ({row['quarter']})**: revenue {self._format_usd(row['revenue_usd'])}, "
                f"operating income {self._format_usd(row['operating_income_usd'])}, "
                f"cash {self._format_usd(row['cash_usd'])}, debt {self._format_usd(row['debt_usd'])}."
            )
            citations.append(f"quarterly_financials — {row['company']}, {row['quarter']}")
        answer = f"Financial summary from the fictional demo database for {quarter}:\n" + (
            "\n".join(lines) if lines else "No records matched the requested period."
        ) + "\n\nThese are synthetic values, not real financial statements or investment advice."
        return AgentAnswer(q, answer, tuple(citations), tuple(trace), not bool(lines))
