import unittest
from src.modeling import choose_threshold, evaluate_classifier

class ModelingTests(unittest.TestCase):
    def test_threshold_selection_returns_candidate_and_report(self):
        threshold, report = choose_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], [0.2, 0.5, 0.8])
        self.assertIn(threshold, report["threshold"].tolist())
        self.assertGreaterEqual(float(report["f1"].max()), 0.0)

    def test_evaluation_contains_auc_and_confusion_matrix(self):
        result = evaluate_classifier([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["confusion_matrix_tn_fp_fn_tp"], [2, 0, 0, 2])

    def test_invalid_probabilities_raise(self):
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [0.1, 1.4])

if __name__ == "__main__":
    unittest.main()
