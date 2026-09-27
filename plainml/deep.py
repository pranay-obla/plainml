"""PyTorch neural networks for tables, wrapped to behave like scikit-learn models.

Optional (``pip install "plainml[torch]"``) and only trained when asked for, with
``--models torch`` or ``--thorough``: on typical spreadsheet data, gradient boosting
usually wins, but a well-regularised network sometimes adds value, especially in an
ensemble. They work anywhere a scikit-learn model does: pipelines, cross-validation,
tuning, ensembles, saving and ``plainml predict``.

The network is a plain multi-layer perceptron: Linear → BatchNorm → ReLU → Dropout
blocks, trained with AdamW, early stopping on a validation split, and the best weights
restored at the end. Training is on the CPU by default for reproducibility; set
``PLAINML_TORCH_DEVICE=cuda`` (or ``mps``) to use a GPU.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _torch() -> Any:
    import torch

    return torch


def _device() -> Any:
    torch = _torch()
    wanted = os.environ.get("PLAINML_TORCH_DEVICE", "cpu").lower()
    if wanted == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        wanted == "mps"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _network(n_inputs: int, n_outputs: int, width: int, depth: int, dropout: float) -> Any:
    nn = _torch().nn
    layers: list[Any] = []
    size = n_inputs
    for level in range(depth):
        hidden = max(8, width // (2**level))
        layers += [nn.Linear(size, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        size = hidden
    layers.append(nn.Linear(size, n_outputs))
    return nn.Sequential(*layers)


class _TorchBase(BaseEstimator):
    def __init__(
        self,
        width: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 15,
        validation_fraction: float = 0.15,
        random_state: int = 42,
    ):
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    # subclasses define: _outputs(), _targets(y), _loss(), _stratify(y)
    def _train(self, X: np.ndarray, targets: np.ndarray, stratify: np.ndarray | None) -> None:
        torch = _torch()
        # One thread: these networks are small, and PyTorch's OpenMP thread pool can deadlock
        # when LightGBM or XGBoost (which ship their own OpenMP) were used in the same process.
        torch.set_num_threads(int(os.environ.get("PLAINML_TORCH_THREADS", "1")))
        torch.manual_seed(self.random_state)
        device = _device()
        X = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        use_validation = len(X) >= 50 and self.validation_fraction > 0
        if use_validation:
            try:
                X_fit, X_val, t_fit, t_val = train_test_split(
                    X,
                    targets,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:  # a class too rare to stratify
                X_fit, X_val, t_fit, t_val = train_test_split(
                    X, targets, test_size=self.validation_fraction, random_state=self.random_state
                )
        else:
            X_fit, t_fit, X_val, t_val = X, targets, X, targets
        network = _network(X.shape[1], self._outputs(), self.width, self.depth, self.dropout).to(
            device
        )
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = self._loss(device)
        X_fit_t = torch.from_numpy(X_fit).to(device)
        t_fit_t = torch.from_numpy(t_fit).to(device)
        X_val_t = torch.from_numpy(X_val).to(device)
        t_val_t = torch.from_numpy(t_val).to(device)
        generator = torch.Generator().manual_seed(self.random_state)
        best_loss, best_state, waited = float("inf"), None, 0
        batch = max(8, min(self.batch_size, len(X_fit)))
        for _epoch in range(self.epochs):
            network.train()
            order = torch.randperm(len(X_fit_t), generator=generator).to(device)
            for start in range(0, len(order), batch):
                index = order[start : start + batch]
                if len(index) < 2:  # batch norm needs at least two rows
                    continue
                optimizer.zero_grad()
                loss = loss_fn(network(X_fit_t[index]), t_fit_t[index])
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                validation = float(loss_fn(network(X_val_t), t_val_t))
            if validation < best_loss - 1e-6:
                best_loss, best_state, waited = validation, copy.deepcopy(network.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        self.network_ = network.to("cpu").eval()
        self.epochs_run_ = _epoch + 1

    def _forward(self, X: Any) -> np.ndarray:
        torch = _torch()
        torch.set_num_threads(int(os.environ.get("PLAINML_TORCH_THREADS", "1")))
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
            return self.network_(tensor).numpy()


class TorchMLPClassifier(ClassifierMixin, _TorchBase):
    """A PyTorch neural network classifier (scikit-learn compatible)."""

    def __init__(
        self,
        width: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 15,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        class_weight: str | None = None,
    ):
        # every parameter is spelled out: scikit-learn's cloning and tuning read the signature
        super().__init__(
            width=width,
            depth=depth,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        self.class_weight = class_weight

    def _outputs(self) -> int:
        return len(self.classes_)

    def _loss(self, device: Any) -> Any:
        torch = _torch()
        weight = None
        if self.class_weight == "balanced":
            weight = torch.tensor(self.class_weights_, dtype=torch.float32, device=device)
        return torch.nn.CrossEntropyLoss(weight=weight)

    def fit(self, X: Any, y: Any) -> TorchMLPClassifier:
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        encoded = self.encoder_.transform(y).astype(np.int64)
        counts = np.bincount(encoded, minlength=len(self.classes_)).astype(float)
        self.class_weights_ = len(encoded) / (len(self.classes_) * np.maximum(counts, 1))
        self._train(X, encoded, encoded)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _TorchBase):
    """A PyTorch neural network regressor (scikit-learn compatible)."""

    def _outputs(self) -> int:
        return self.n_targets_

    def _loss(self, device: Any) -> Any:
        return _torch().nn.HuberLoss(delta=1.0)

    def fit(self, X: Any, y: Any) -> TorchMLPRegressor:
        values = np.asarray(y, dtype=np.float32)
        self.single_target_ = values.ndim == 1
        values = values.reshape(len(values), -1)
        self.n_targets_ = values.shape[1]
        self.y_mean_ = values.mean(axis=0)
        self.y_scale_ = values.std(axis=0) + 1e-8  # train on standardised targets
        self._train(X, ((values - self.y_mean_) / self.y_scale_).astype(np.float32), None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        values = self._forward(X) * self.y_scale_ + self.y_mean_
        return values.ravel() if self.single_target_ else values
