"""CLI for the educational Healthcare RAG prototype."""
import argparse
import json
from pathlib import Path
from .health_rag import HealthcareRAG


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a local educational healthcare document collection.")
    parser.add_argument("query", help="Question to retrieve evidence for")
    parser.add_argument("--data-dir", default="data/knowledge_base")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--strategy", choices=("heading", "fixed"), default="heading")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output")
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"})
    if not paths:
        parser.error(f"No supported documents found under {args.data_dir}")
    rag = HealthcareRAG.from_paths(paths, strategy=args.strategy)
    result = rag.answer(args.query, top_k=args.top_k)
    payload = {
        "query": result.query, "answer": result.answer, "citations": result.citations,
        "abstained": result.abstained,
        "retrieved": [{"source": r.chunk.source, "chunk_id": r.chunk.chunk_id,
                       "section": r.chunk.section, "page": r.chunk.page, "score": round(r.score, 4)}
                      for r in result.retrieved],
        "safety_note": "Educational retrieval only; not a diagnostic or treatment recommendation.",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
