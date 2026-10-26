import unittest
from pathlib import Path
from src.health_rag import DocumentChunk, HealthcareRAG, LocalTfidfIndex, SourcePage, chunk_sources, load_source_pages
from src.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "knowledge_base"


class ChunkingTests(unittest.TestCase):
    def test_heading_chunking_preserves_markdown_sections(self):
        pages = [SourcePage("guide.md", "# First\nalpha text\n\n## Second\nbeta text")]
        chunks = chunk_sources(pages, strategy="heading", chunk_size=200, overlap=20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c.section for c in chunks], ["First", "Second"])

    def test_fixed_chunking_creates_overlapping_chunks(self):
        text = " ".join(["clinical" for _ in range(100)])
        chunks = chunk_sources([SourcePage("guide.txt", text)], strategy="fixed", chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.text for c in chunks))

    def test_invalid_strategy_fails(self):
        with self.assertRaises(ValueError):
            chunk_sources([SourcePage("a.txt", "text")], strategy="unknown")

    def test_unsupported_file_type_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("x,y", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_source_pages(path)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = sorted(DATA.glob("*.md"))
        cls.rag = HealthcareRAG.from_paths(cls.paths)

    def test_index_rejects_empty_chunk_list(self):
        with self.assertRaises(ValueError):
            LocalTfidfIndex([])

    def test_query_rejects_blank(self):
        with self.assertRaises(ValueError):
            self.rag.retrieve("   ")

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.rag.retrieve("medicine", top_k=0)

    def test_appointment_query_retrieves_expected_document(self):
        results = self.rag.retrieve("What information should I prepare before an appointment?", top_k=3)
        self.assertEqual(results[0].chunk.source, "appointment_preparation.md")

    def test_medicine_query_retrieves_safety_document(self):
        results = self.rag.retrieve("What should I do before taking an unfamiliar medicine?", top_k=3)
        self.assertEqual(results[0].chunk.source, "medication_safety.md")

    def test_answer_includes_citation(self):
        answer = self.rag.answer("What should I bring to an appointment?")
        self.assertFalse(answer.abstained)
        self.assertTrue(answer.citations)
        self.assertIn("appointment_preparation.md", answer.answer)

    def test_answer_prefers_relevant_evidence_over_document_prefaces(self):
        answer = self.rag.answer("What information should I prepare before an appointment?")
        self.assertIn("write down your main questions", answer.answer.lower())
        self.assertNotIn("fictional document is included", answer.answer.lower())

    def test_unrelated_query_abstains(self):
        answer = HealthcareRAG.from_paths(self.paths, min_score=0.95).answer("quantum engine torque warranty")
        self.assertTrue(answer.abstained)
        self.assertFalse(answer.citations)

    def test_urgent_query_retrieves_urgent_guidance(self):
        results = self.rag.retrieve("What for severe breathing difficulty?", top_k=3)
        self.assertEqual(results[0].chunk.source, "urgent_care_guidance.md")

    def test_retrieval_evaluation_runs(self):
        report = evaluate(DATA)
        self.assertEqual(report["cases"], 3)
        self.assertGreaterEqual(report["recall_at_k"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
