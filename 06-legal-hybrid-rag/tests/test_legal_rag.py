import tempfile
import unittest
from pathlib import Path

from src.legal_rag import BM25Index, Clause, LegalHybridRAG, TfidfVectorIndex, load_contract, load_corpus
from src.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "contracts"
PATHS = sorted(DATA.glob("*.md"))


class IngestionTests(unittest.TestCase):
    def test_loads_numbered_clauses(self):
        clauses = load_contract(DATA / "fictional_saas_agreement.md")
        self.assertGreaterEqual(len(clauses), 6)
        self.assertTrue(any(c.clause_id == "4.1" for c in clauses))

    def test_heading_title_is_separated_from_clause_id(self):
        clauses = load_contract(DATA / "fictional_saas_agreement.md")
        clause = next(c for c in clauses if c.clause_id == "4.1")
        self.assertEqual(clause.title, "Fees and Invoicing")

    def test_plain_text_without_heading_becomes_one_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.txt"
            path.write_text("Confidentiality and payment are described here.", encoding="utf-8")
            clauses = load_contract(path)
            self.assertEqual(len(clauses), 1)
            self.assertEqual(clauses[0].clause_id, "S01")

    def test_unsupported_file_type_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.pdf"
            path.write_text("not a real pdf", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_contract(path)

    def test_missing_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_contract(DATA / "does_not_exist.md")

    def test_load_corpus_requires_documents(self):
        with self.assertRaises(ValueError):
            load_corpus([])


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clauses = load_corpus(PATHS)
        cls.bm25 = BM25Index(cls.clauses)
        cls.vector = TfidfVectorIndex(cls.clauses)
        cls.rag = LegalHybridRAG(cls.clauses)

    def test_bm25_finds_payment_clause(self):
        ranked = self.bm25.rank("When are subscription invoices due for payment?")
        self.assertEqual(self.clauses[ranked[0][0]].clause_id, "4.1")

    def test_vector_search_finds_payment_clause(self):
        ranked = self.vector.rank("When are subscription invoices due for payment?")
        self.assertEqual(self.clauses[ranked[0][0]].clause_id, "4.1")

    def test_hybrid_finds_termination_clause(self):
        results = self.rag.retrieve("When may either party terminate for a material breach?", top_k=3)
        self.assertEqual(results[0].clause.clause_id, "7.1")

    def test_answer_contains_clause_citation(self):
        answer = self.rag.answer("When are subscription invoices due for payment?")
        self.assertFalse(answer.abstained)
        self.assertIn("Clause 4.1", answer.answer)
        self.assertIn("fictional_saas_agreement.md", answer.citations[0])

    def test_unmatched_query_abstains(self):
        answer = self.rag.answer("quantum flux lunar telescope")
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.citations, ())

    def test_empty_index_rejected(self):
        with self.assertRaises(ValueError):
            LegalHybridRAG([])

    def test_blank_query_rejected(self):
        with self.assertRaises(ValueError):
            self.rag.retrieve("  ")

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.rag.retrieve("invoice due", top_k=0)

    def test_evaluation_runs_and_finds_relevant_clauses(self):
        report = evaluate(DATA, top_k=5)
        self.assertEqual(report["cases"], 6)
        self.assertGreaterEqual(report["recall_at_5"], 0.8)
        self.assertGreaterEqual(report["mrr_at_5"], 0.6)


if __name__ == "__main__":
    unittest.main()
