"""Controlled local tools for the enterprise logistics RAG demo."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from .ingestion import DEFAULT_SHIPMENT_FILE, load_shipments
from .models import ToolResult

SHIPMENT_ID_RE = re.compile(r"\bSHP-\d{4,6}\b", re.I)


class ShipmentLookupTool:
    """Read-only lookup over a small synthetic shipment events JSONL."""
    name = "shipment_lookup"

    def __init__(self, data_path: str | Path = DEFAULT_SHIPMENT_FILE) -> None:
        self.data_path = Path(data_path)

    def lookup(self, shipment_id: str) -> ToolResult:
        candidate = shipment_id.strip().upper()
        if not SHIPMENT_ID_RE.fullmatch(candidate):
            return ToolResult(self.name, False, "Invalid synthetic shipment ID format")
        records = load_shipments(self.data_path)
        for record in records:
            if record["shipment_id"] == candidate:
                return ToolResult(
                    self.name, True, "Found one synthetic shipment record",
                    {"shipment": record},
                    f"shipment_events.jsonl#{candidate}",
                )
        return ToolResult(
            self.name, False, "Shipment ID not found in the local synthetic dataset",
            {"shipment_id": candidate}, None,
        )


def is_tracking_intent(question: str) -> bool:
    lowered = question.casefold()
    verbs = ("track", "tracking", "shipment status", "status of", "where is", "delivery status", "has shipment")
    return any(verb in lowered for verb in verbs)
