"""Command-line interface for the local enterprise agentic RAG demo."""
from __future__ import annotations
import argparse
import json
import sys
from .agent import EnterpriseRAGAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the synthetic logistics knowledge base and shipment records.")
    parser.add_argument("question", help="Logistics policy question or synthetic shipment tracking question")
    parser.add_argument("--json", action="store_true", help="Emit structured answer, citations, safety decision and trace")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 10:
        parser.error("--top-k must be between 1 and 10")
    try:
        response = EnterpriseRAGAgent(top_k=args.top_k).answer(args.question)
    except (OSError, ValueError) as exc:
        print(f"Agent failed: {exc}", file=sys.stderr)
        return 2
    payload = response.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(response.answer)
        if response.abstained:
            print("\n[Abstained: evidence was missing or the request was blocked.]")
        if response.citations:
            print("\nCitations: " + ", ".join(response.citations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
