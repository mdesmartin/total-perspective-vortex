"""
csp.py — Common Spatial Patterns as a scikit-learn transformer.

Public API
----------
CSP
    Binary spatial-filter transformer.
    fit() learns the projection W; transform() returns log-variance features
    of shape (n_epochs, n_components).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Average trace-normalized covariance over a set of epochs.

    For each epoch X of shape (n_channels, n_times):

        C = (X_c X_c^T) / trace(X_c X_c^T)

    where X_c is X with the per-channel mean subtracted. Averaging these
    matrices across epochs gives one class covariance of shape
    (n_channels, n_channels).
    """
    x = epochs - epochs.mean(axis=2, keepdims=True)
    covs = np.matmul(x, x.transpose(0, 2, 1))
    traces = np.trace(covs, axis1=1, axis2=2)
    traces = np.maximum(traces, _EPS)
    covs /= traces[:, np.newaxis, np.newaxis]
    return covs.mean(axis=0)


def _whitening_matrix(C: np.ndarray) -> np.ndarray:
    """Return P such that P @ C @ P.T ≈ I.

    C = U Λ U^T  ⇒  P = Λ^{-1/2} U^T
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, _EPS)
    return (eigvecs / np.sqrt(eigvals)).T


def _pick_component_indices(n_components: int, n_channels: int) -> np.ndarray:
    """Top n//2 and bottom n - n//2 eigenvectors (eigenvalues already descending)."""
    n_head = n_components // 2
    n_tail = n_components - n_head
    return np.concatenate(
        [
            np.arange(n_head),
            np.arange(n_channels - n_tail, n_channels),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns for binary EEG classification.

    Finds spatial filters W that maximise the variance of one class while
    minimising the variance of the other. After projection, each epoch is
    reduced to a log-variance feature vector of length ``n_components``.

    Parameters
    ----------
    n_components : int, default=4
        Number of spatial filters kept. The first half correspond to the
        largest eigenvalues (high variance for class 1) and the second half
        to the smallest (high variance for class 2). An even number is
        recommended.

    Attributes
    ----------
    filters_ : ndarray of shape (n_components, n_channels)
        Spatial filters W, one filter per row. Projection of one epoch is
        ``filters_ @ epoch``.
    eigenvalues_ : ndarray of shape (n_components,)
        Generalised eigenvalues of the selected filters (in [0, 1] after
        whitening). Close to 1 → discriminative for class 1; close to 0 →
        discriminative for class 2.
    classes_ : ndarray of shape (2,)
        Class labels in the order used to build C1 then C2.
    n_channels_ : int
        Number of channels seen during :meth:`fit`.
    """

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y):
        """Learn spatial filters from labelled epochs.

        Parameters
        ----------
        X : ndarray of shape (n_epochs, n_channels, n_times)
            Band-pass filtered EEG epochs.
        y : ndarray of shape (n_epochs,)
            Integer class labels. Must contain exactly two distinct values.

        Returns
        -------
        self : CSP
        """
        X = _validate_epochs(X)
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X.shape[0]} epochs vs {y.shape[0]} labels"
            )

        n_epochs, n_channels, _n_times = X.shape
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError(f"n_components must be a positive int, got {self.n_components}")
        if self.n_components > n_channels:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_channels={n_channels}"
            )

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {classes.size}: {classes}"
            )
        for cls in classes:
            if np.sum(y == cls) < 1:
                raise ValueError(f"class {cls} has no epochs")

        # 1. Class covariances C1, C2
        C1 = _mean_normalized_covariance(X[y == classes[0]])
        C2 = _mean_normalized_covariance(X[y == classes[1]])

        # 2–3. Composite covariance and whitening matrix P
        P = _whitening_matrix(C1 + C2)

        # 4–5. Eigendecompose whitened C1 (S1 = P C1 P^T)
        #      After whitening, eigenvalues of S2 are 1 - eigenvalues of S1.
        S1 = P @ C1 @ P.T
        eigvals, eigvecs = np.linalg.eigh(S1)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # 6. W = U^T P, then keep top + bottom filters
        W_full = eigvecs.T @ P
        idx = _pick_component_indices(self.n_components, n_channels)

        self.filters_ = W_full[idx]
        self.eigenvalues_ = eigvals[idx]
        self.classes_ = classes
        self.n_channels_ = n_channels
        return self

    def transform(self, X):
        """Project epochs and return log-variance features.

        For each epoch the signal is spatially filtered, ``Z = W X``, then

            f_i = log( var(Z_i) / sum_j var(Z_j) )

        Parameters
        ----------
        X : ndarray of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        features : ndarray of shape (n_epochs, n_components)
        """
        check_is_fitted(self, "filters_")
        X = _validate_epochs(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, expected {self.n_channels_}"
            )

        # (n_components, n_channels) @ (n_epochs, n_channels, n_times)
        # → (n_epochs, n_components, n_times)
        Z = np.matmul(self.filters_, X)
        var = np.var(Z, axis=2)
        var = np.maximum(var, _EPS)
        var /= var.sum(axis=1, keepdims=True)
        return np.log(var)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        tags.target_tags.required = True
        return tags
