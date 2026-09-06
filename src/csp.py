"""Common Spatial Patterns as a scikit-learn transformer."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_EPS = 1e-12


def _validate_epochs(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_epochs, n_channels, n_times), "
            f"got ndim={X.ndim} shape={X.shape}"
        )
    if X.shape[0] == 0:
        raise ValueError(f"{name} contains no epochs")
    return X


def _mean_normalized_covariance(epochs: np.ndarray) -> np.ndarray:
    """Average trace-normalized covariance over epochs → (n_channels, n_channels)."""
    x = epochs - epochs.mean(axis=2, keepdims=True)
    covs = np.matmul(x, x.transpose(0, 2, 1))
    traces = np.trace(covs, axis1=1, axis2=2)
    traces = np.maximum(traces, _EPS)
    covs /= traces[:, np.newaxis, np.newaxis]
    return covs.mean(axis=0)


def _whitening_matrix(C: np.ndarray) -> np.ndarray:
    """P such that P @ C @ P.T ≈ I  (C = U Λ U^T ⇒ P = Λ^{-1/2} U^T)."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, _EPS)
    return (eigvecs / np.sqrt(eigvals)).T


def _pick_component_indices(n_components: int, n_channels: int) -> np.ndarray:
    """Top n//2 and bottom n - n//2 eigenvectors (eigenvalues descending)."""
    n_head = n_components // 2
    n_tail = n_components - n_head
    return np.concatenate(
        [
            np.arange(n_head),
            np.arange(n_channels - n_tail, n_channels),
        ]
    )


def _check_fit_inputs(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (X, y, classes, n_channels) after shape / binary-label checks."""
    X = _validate_epochs(X)
    y = np.asarray(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y length mismatch: {X.shape[0]} epochs vs {y.shape[0]} labels"
        )

    n_channels = X.shape[1]
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError(f"n_components must be a positive int, got {n_components}")
    if n_components > n_channels:
        raise ValueError(
            f"n_components={n_components} exceeds n_channels={n_channels}"
        )

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            f"CSP requires exactly 2 classes, got {classes.size}: {classes}"
        )
    for cls in classes:
        if np.sum(y == cls) < 1:
            raise ValueError(f"class {cls} has no epochs")

    return X, y, classes, n_channels


def _class_covariances(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    C1 = _mean_normalized_covariance(X[y == classes[0]])
    C2 = _mean_normalized_covariance(X[y == classes[1]])
    return C1, C2


def _solve_spatial_filters(
    C1: np.ndarray,
    C2: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten C1+C2, eigendecompose whitened C1, keep extreme filters."""
    n_channels = C1.shape[0]
    P = _whitening_matrix(C1 + C2)
    S1 = P @ C1 @ P.T
    eigvals, eigvecs = np.linalg.eigh(S1)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W_full = eigvecs.T @ P
    idx = _pick_component_indices(n_components, n_channels)
    return W_full[idx], eigvals[idx]


def _log_variance_features(filters: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project epochs with W then return normalised log-variance features."""
    Z = np.matmul(filters, X)
    var = np.var(Z, axis=2)
    var = np.maximum(var, _EPS)
    var /= var.sum(axis=1, keepdims=True)
    return np.log(var)


class CSP(BaseEstimator, TransformerMixin):
    """Binary CSP: fit spatial filters W, transform epochs to log-variance features."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y):
        """Learn filters that maximise class-1 variance / minimise class-2 variance."""
        X, y, classes, n_channels = _check_fit_inputs(X, y, self.n_components)
        C1, C2 = _class_covariances(X, y, classes)
        filters, eigenvalues = _solve_spatial_filters(C1, C2, self.n_components)

        self.filters_ = filters
        self.eigenvalues_ = eigenvalues
        self.classes_ = classes
        self.n_channels_ = n_channels
        return self

    def transform(self, X):
        """Project epochs and return log-variance features (n_epochs, n_components)."""
        check_is_fitted(self, "filters_")
        X = _validate_epochs(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, expected {self.n_channels_}"
            )
        return _log_variance_features(self.filters_, X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        tags.target_tags.required = True
        return tags
