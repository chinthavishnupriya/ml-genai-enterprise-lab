"""Offline ingestion for a local logistics-policy corpus and shipment JSONL."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from .models import KnowledgeChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_DIR = PROJECT_ROOT / "data" / "knowledge"
DEFAULT_SHIPMENT_FILE = PROJECT_ROOT / "data" / "shipment_events.jsonl"
REQUIRED_SHIPMENT_FIELDS = {
    "shipment_id", "status", "origin", "destination", "estimated_delivery",
    "last_update", "update_note", "carrier",
}


def load_policy_chunks(policy_dir: str | Path = DEFAULT_POLICY_DIR) -> list[KnowledgeChunk]:
    directory = Path(policy_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    chunks: list[KnowledgeChunk] = []
    for path in sorted(directory.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        current_heading = path.stem.replace("_", " ").title()
        body: list[str] = []
        def flush() -> None:
            text = " ".join(line.strip() for line in body if line.strip()).strip()
            if text:
                chunks.append(KnowledgeChunk(
                    source=path.name,
                    section=current_heading,
                    text=text,
                    chunk_id=f"{path.name}:{len(chunks)+1}",
                ))
        for line in raw.splitlines():
            match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
            if match:
                flush()
                current_heading = match.group(1).strip()
                body = []
            else:
                body.append(line)
        flush()
    return chunks


def load_shipments(path: str | Path = DEFAULT_SHIPMENT_FILE) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict) or not REQUIRED_SHIPMENT_FIELDS.issubset(row):
                raise ValueError(f"Shipment at line {line_number} is missing required fields")
            shipment_id = row["shipment_id"]
            if not isinstance(shipment_id, str) or not re.fullmatch(r"SHP-\d{4,6}", shipment_id):
                raise ValueError(f"Invalid synthetic shipment ID at line {line_number}")
            if shipment_id in seen:
                raise ValueError(f"Duplicate shipment ID at line {line_number}: {shipment_id}")
            seen.add(shipment_id)
            if not all(isinstance(row[field], str) and row[field].strip() for field in REQUIRED_SHIPMENT_FIELDS):
                raise ValueError(f"Invalid shipment field type at line {line_number}")
            rows.append(row)
    return rows


def build_corpus(policy_dir: str | Path = DEFAULT_POLICY_DIR) -> list[KnowledgeChunk]:
    """Return policy chunks. Operational shipment data is exposed by a separate lookup tool."""
    chunks = load_policy_chunks(policy_dir)
    if not chunks:
        raise ValueError("No non-empty policy sections were found")
    return chunks


def write_corpus_manifest(output_path: str | Path, chunks: list[KnowledgeChunk] | None = None) -> dict[str, Any]:
    """Write safe metadata only, not private source data."""
    from .models import KnowledgeChunk
    chunks = chunks if chunks is not None else build_corpus()
    manifest = {
        "corpus_type": "synthetic logistics policy knowledge base",
        "chunk_count": len(chunks),
        "sources": sorted({chunk.source for chunk in chunks}),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "disclaimer": "Fictional training corpus; not a live carrier policy or operational source.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    rows = load_shipments()
    chunks = build_corpus()
    manifest = write_corpus_manifest(PROJECT_ROOT / "data" / "processed" / "corpus_manifest.json", chunks)
    print(json.dumps({
        "shipments_loaded": len(rows),
        "policy_chunks_loaded": len(chunks),
        "policy_sources": manifest["sources"],
        "manifest": "data/processed/corpus_manifest.json",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
