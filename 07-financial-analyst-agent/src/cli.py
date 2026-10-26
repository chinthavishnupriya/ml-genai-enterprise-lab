"""CLI for the offline Financial Analyst Agent."""
import argparse
import json
from dataclasses import asdict
from .agent import FinancialAnalystAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze synthetic financial data. Not financial advice.")
    parser.add_argument("question", help="Question about sample metrics or fictional research notes")
    parser.add_argument("--json", action="store_true", help="Emit structured answer and agent trace")
    parser.add_argument("--db", default=None, help="Optional SQLite database path")
    args = parser.parse_args()
    agent = FinancialAnalystAgent(database_path=args.db) if args.db else FinancialAnalystAgent()
    result = agent.answer(args.question)
    payload = {
        "question": result.question, "answer": result.answer,
        "citations": result.citations, "abstained": result.abstained,
        "trace": [asdict(event) for event in result.trace],
        "disclaimer": "Synthetic financial analysis demonstration only; not investment, tax, or financial advice."
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
