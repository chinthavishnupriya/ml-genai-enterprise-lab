"""Small deterministic retrieval evaluation for the included synthetic corpus."""
from pathlib import Path
from .health_rag import HealthcareRAG

CASES = [
    ("What information should I prepare before an appointment?", "appointment_preparation.md"),
    ("What should I do before taking an unfamiliar medicine?", "medication_safety.md"),
    ("What should someone do for severe breathing difficulty?", "urgent_care_guidance.md"),
]


def evaluate(data_dir: str | Path = "data/knowledge_base", top_k: int = 3) -> dict:
    paths = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"})
    rag = HealthcareRAG.from_paths(paths)
    hits = 0
    reciprocal_ranks = []
    rows = []
    for query, expected_source in CASES:
        results = rag.retrieve(query, top_k=top_k)
        rank = next((i + 1 for i, item in enumerate(results) if item.chunk.source == expected_source), None)
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        rows.append({"query": query, "expected_source": expected_source, "rank": rank,
                     "top_source": results[0].chunk.source if results else None})
    count = len(CASES)
    return {"cases": count, "recall_at_k": round(hits / count, 4),
            "mrr_at_k": round(sum(reciprocal_ranks) / count, 4), "k": top_k, "rows": rows,
            "note": "Tiny hand-authored retrieval smoke set for the synthetic corpus; not clinical validation."}


def main() -> int:
    import json
    print(json.dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
