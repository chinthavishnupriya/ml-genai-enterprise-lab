import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion import build_corpus, load_policy_chunks, load_shipments
from src.retrieval import HybridRetriever


class IngestionTests(unittest.TestCase):
    def test_loads_seven_unique_synthetic_shipments(self):
        rows = load_shipments()
        self.assertEqual(len(rows), 7)
        self.assertEqual(len({row["shipment_id"] for row in rows}), 7)

    def test_loads_multiple_policy_sections(self):
        chunks = build_corpus()
        self.assertGreaterEqual(len(chunks), 10)
        self.assertIn("cross_border.md", {chunk.source for chunk in chunks})

    def test_rejects_malformed_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shipments(path)

    def test_rejects_duplicate_shipment_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            row = {
                "shipment_id":"SHP-1001","status":"In transit","origin":"A","destination":"B",
                "estimated_delivery":"unknown","last_update":"synthetic","update_note":"none","carrier":"demo"
            }
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shipments(path)

    def test_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps({"shipment_id":"SHP-1001"})+"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shipments(path)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(build_corpus())

    def test_delay_query_finds_delay_policy(self):
        hits = self.retriever.search("What should operations do when a shipment is delayed?", top_k=3)
        self.assertTrue(hits)
        self.assertIn("delay_escalation.md", {hit.chunk.source for hit in hits})

    def test_cross_border_query_finds_customs_policy(self):
        hits = self.retriever.search("Which documents are required for international customs clearance?", top_k=3)
        self.assertIn("cross_border.md", {hit.chunk.source for hit in hits})

    def test_cross_border_escalation_query_finds_exact_section(self):
        hits = self.retriever.search("How should cross-border delays be escalated?", top_k=2)
        citations = {hit.chunk.citation for hit in hits}
        self.assertIn("cross_border.md#Escalation and holds", citations)

    def test_delivered_status_query_finds_exact_section(self):
        hits = self.retriever.search("What does delivered status mean in the tracking data?", top_k=2)
        citations = {hit.chunk.citation for hit in hits}
        self.assertIn("shipment_status.md#Delivered", citations)

    def test_temperature_query_finds_temperature_policy(self):
        hits = self.retriever.search("What precautions are required for temperature-controlled shipments?", top_k=3)
        self.assertIn("temperature_control.md", {hit.chunk.source for hit in hits})

    def test_no_match_returns_empty(self):
        self.assertEqual(self.retriever.search("quantum rocket telescope"), [])

    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("shipment", top_k=0)


if __name__ == "__main__":
    unittest.main()
