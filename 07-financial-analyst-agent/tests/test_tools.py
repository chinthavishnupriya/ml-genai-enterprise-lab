import tempfile
import unittest
from pathlib import Path

from src.tools import LocalDocumentSearchTool, ReadOnlySQLTool, SafeCalculator


class ReadOnlySQLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sample.sqlite"
        self.tool = ReadOnlySQLTool(self.db, max_rows=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_synthetic_rows(self):
        result = self.tool.query("SELECT company, quarter FROM quarterly_financials ORDER BY company, quarter")
        self.assertEqual(len(result.data["rows"]), 2)
        self.assertTrue(result.data["truncated"])

    def test_parameterized_query(self):
        result = self.tool.query("SELECT company, revenue_usd FROM quarterly_financials WHERE company = ? AND quarter = ?", ("ACME Systems", "2025-Q4"))
        self.assertEqual(len(result.data["rows"]), 1)
        self.assertEqual(result.data["rows"][0]["revenue_usd"], 1400000)

    def test_rejects_delete(self):
        with self.assertRaises(ValueError):
            self.tool.query("DELETE FROM quarterly_financials")

    def test_rejects_multistatement(self):
        with self.assertRaises(ValueError):
            self.tool.query("SELECT * FROM quarterly_financials; SELECT * FROM quarterly_financials")

    def test_rejects_sql_comments(self):
        with self.assertRaises(ValueError):
            self.tool.query("SELECT * FROM quarterly_financials -- bypass")

    def test_rejects_unapproved_table(self):
        with self.assertRaises(ValueError):
            self.tool.query("SELECT name FROM sqlite_master")

    def test_rejects_empty_sql(self):
        with self.assertRaises(ValueError):
            self.tool.query("")


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.calc = SafeCalculator()

    def test_calculates_operating_margin(self):
        value = self.calc.calculate("210000 / 1400000 * 100").data["value"]
        self.assertAlmostEqual(value, 15.0)

    def test_rejects_python_calls(self):
        with self.assertRaises(ValueError):
            self.calc.calculate("__import__('os').system('echo bad')")

    def test_rejects_division_by_zero(self):
        with self.assertRaises(ValueError):
            self.calc.calculate("10 / 0")

    def test_rejects_large_exponent(self):
        with self.assertRaises(ValueError):
            self.calc.calculate("2 ** 100")


class DocumentSearchTests(unittest.TestCase):
    def test_search_returns_relevant_source_citation(self):
        doc_dir = Path(__file__).resolve().parents[1] / "data" / "research"
        result = LocalDocumentSearchTool(doc_dir).search("liquidity and leverage risks")
        self.assertTrue(result.data)
        self.assertEqual(result.data[0]["source"], "risk_register.md")
        self.assertIn("Liquidity and leverage", result.data[0]["section"])

    def test_rejects_blank_search(self):
        tool = LocalDocumentSearchTool(Path("."))
        with self.assertRaises(ValueError):
            tool.search(" ")


if __name__ == "__main__":
    unittest.main()
