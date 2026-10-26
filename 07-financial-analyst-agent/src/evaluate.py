"""Smoke evaluation for tool routing and evidence/citation behavior."""
from .agent import FinancialAnalystAgent

CASES = [
    ("What was ACME Systems operating margin in Q4?", "ACME Systems", "margin"),
    ("Compare Q3 to Q4 revenue growth across the companies.", "Cobalt Manufacturing", "growth"),
    ("What are the main liquidity and leverage risks?", "risk_register.md", "risk"),
    ("Show Q4 cash and debt for the companies.", "2025-Q4", "cash"),
]


def evaluate() -> dict:
    agent = FinancialAnalystAgent()
    results = []
    for question, expected_fragment, _ in CASES:
        answer = agent.answer(question)
        expected = expected_fragment.casefold() in (answer.answer + " " + " ".join(answer.citations)).casefold()
        results.append({
            "question": question, "expected_fragment": expected_fragment,
            "matched": expected, "trace_steps": len(answer.trace),
            "citations": list(answer.citations), "abstained": answer.abstained
        })
    n = len(results)
    return {
        "cases": n,
        "answer_coverage": round(sum(row["matched"] for row in results) / n, 4),
        "results": results,
        "note": "Hand-authored synthetic smoke set. It checks tool routing/citation paths, not financial correctness."
    }


def main() -> int:
    print(__import__("json").dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
