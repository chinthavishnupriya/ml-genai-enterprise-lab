import unittest
import numpy as np
import torch
from src.modeling import DemandMLP, make_search_configs, regression_metrics

class DemandModelingTests(unittest.TestCase):
    def test_search_is_reproducible(self):
        self.assertEqual(make_search_configs(42, 4), make_search_configs(42, 4))

    def test_mlp_output_shape(self):
        model = DemandMLP(input_dim=12, hidden_layers=(16, 8), dropout=0.1)
        output = model(torch.zeros((5, 12)))
        self.assertEqual(tuple(output.shape), (5,))

    def test_metrics_are_finite(self):
        result = regression_metrics([10, 12, 14], [11, 11, 15])
        self.assertTrue(np.isfinite(result["rmse"]))
        self.assertTrue(np.isfinite(result["mae"]))
        self.assertTrue(0 <= result["smape_percent"] <= 200)

    def test_bad_prediction_shape_raises(self):
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [1])

if __name__ == "__main__":
    unittest.main()
