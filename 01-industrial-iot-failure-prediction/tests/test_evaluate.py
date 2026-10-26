import unittest
from src.evaluate import threshold_report, classification_metrics

class EvaluationTests(unittest.TestCase):
    def test_threshold_report_selects_valid_threshold(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.3, 0.7, 0.9]
        report, threshold = threshold_report(y, p, thresholds=[0.2, 0.5, 0.8], min_precision=0.5)
        self.assertIn(threshold, report["threshold"].tolist())
        self.assertGreaterEqual(len(report), 1)

    def test_metrics_has_confusion_matrix(self):
        result = classification_metrics([0, 1, 1], [0.1, 0.7, 0.2], 0.5)
        self.assertEqual(len(result["confusion_matrix_tn_fp_fn_tp"]), 4)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_invalid_probabilities_raise(self):
        import unittest
        with self.assertRaises(ValueError):
            threshold_report([0, 1], [0.1, 1.5])

if __name__ == "__main__":
    unittest.main()
