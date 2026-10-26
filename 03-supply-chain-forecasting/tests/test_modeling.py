import unittest
import numpy as np
import torch
from src.modeling import DemandMLP, make_search_configs, regression_metrics, train_trial

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

    def test_training_is_reproducible_for_fixed_seed(self):
        x = np.arange(80, dtype=float).reshape(20, 4) / 80.0
        y = np.linspace(0.1, 1.2, 20)
        config = {"hidden_layers": [8], "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0}
        first, _, _ = train_trial(x, y, x[:6], y[:6], config, max_epochs=2, patience=2, seed=123)
        second, _, _ = train_trial(x, y, x[:6], y[:6], config, max_epochs=2, patience=2, seed=123)
        first.eval()
        second.eval()
        with torch.no_grad():
            p1 = first(torch.as_tensor(x[:6], dtype=torch.float32)).numpy()
            p2 = second(torch.as_tensor(x[:6], dtype=torch.float32)).numpy()
        np.testing.assert_allclose(p1, p2, rtol=1e-6, atol=1e-6)

    def test_bad_prediction_shape_raises(self):
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [1])

if __name__ == "__main__":
    unittest.main()
