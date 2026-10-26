"""Preprocessing, PyTorch MLP, random-search configs and regression metrics."""
from __future__ import annotations
import copy
import itertools
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


class DemandMLP(nn.Module):
    """Dense feedforward regressor with BatchNorm and Dropout per hidden layer."""
    def __init__(self, input_dim: int, hidden_layers=(64, 32), dropout: float = 0.15):
        super().__init__()
        layers = []
        in_features = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)])
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def make_search_configs(seed: int = 42, n_trials: int = 6):
    """Sample diverse deterministic configs from a small random-search space."""
    hidden_choices = [(64, 32), (64, 64), (128, 64), (128, 64, 32)]
    dropout_choices = [0.10, 0.20, 0.30]
    lr_choices = [0.001, 0.003]
    wd_choices = [0.0, 1e-5, 1e-4]
    space = [
        {"hidden_layers": list(hidden), "dropout": dropout, "learning_rate": lr, "weight_decay": wd}
        for hidden, dropout, lr, wd in itertools.product(hidden_choices, dropout_choices, lr_choices, wd_choices)
    ]
    if n_trials < 1 or n_trials > len(space):
        raise ValueError(f"n_trials must be between 1 and {len(space)}")
    return random.Random(seed).sample(space, n_trials)


def _tensors(x, y=None):
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if y is None:
        return xt
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    return xt, yt


def _train_epochs(model, x_train, y_train, epochs, learning_rate, weight_decay, seed=42,
                  x_validation=None, y_validation=None, patience=8):
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = model.to(torch.device("cpu"))
    xt, yt = _tensors(x_train, y_train)
    dataset = TensorDataset(xt, yt)
    batch_size = min(512, len(dataset))
    drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    if x_validation is not None:
        xv, yv = _tensors(x_validation, y_validation)
    history = {"epoch": [], "training_loss_log_space": [], "validation_loss_log_space": []}
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
        history["epoch"].append(epoch)
        history["training_loss_log_space"].append(train_loss)
        if x_validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(xv), yv).cpu())
            history["validation_loss_log_space"].append(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_trial(x_train, y_train_log, x_val, y_val_log, config, max_epochs=45, patience=8, seed=42):
    torch.manual_seed(seed)  # Seed parameter initialization as well as training.
    model = DemandMLP(
        input_dim=np.asarray(x_train).shape[1],
        hidden_layers=tuple(config["hidden_layers"]),
        dropout=float(config["dropout"]),
    )
    return _train_epochs(model, x_train, y_train_log, max_epochs,
                         float(config["learning_rate"]), float(config["weight_decay"]),
                         seed=seed, x_validation=x_val, y_validation=y_val_log, patience=patience)


def fit_final_model(x, y_log, config, epochs, seed=42):
    torch.manual_seed(seed)  # Reproducible final-model initialization.
    model = DemandMLP(input_dim=np.asarray(x).shape[1],
                      hidden_layers=tuple(config["hidden_layers"]), dropout=float(config["dropout"]))
    model, history, _ = _train_epochs(model, x, y_log, max(1, int(epochs)),
                                      float(config["learning_rate"]), float(config["weight_decay"]), seed=seed)
    return model, history


def predict_demand(model: DemandMLP, x) -> np.ndarray:
    model.eval()
    xt = _tensors(x)
    with torch.no_grad():
        log_predictions = model(xt).cpu().numpy()
    return np.maximum(0.0, np.expm1(np.clip(log_predictions, -20, 20)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted arrays must have equal, non-zero length.")
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)) * 100)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "smape_percent": smape,
    }
