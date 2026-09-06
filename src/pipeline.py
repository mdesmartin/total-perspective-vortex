"""sklearn Pipeline factory: CSP → LDA."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .csp import CSP

DEFAULT_N_COMPONENTS = 6


def make_pipeline(
    n_components: int = DEFAULT_N_COMPONENTS,
    clf: ClassifierMixin | None = None,
) -> Pipeline:
    """Build Pipeline([CSP, classifier]); LDA by default."""
    if clf is None:
        clf = LinearDiscriminantAnalysis()
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components)),
            ("clf", clf),
        ]
    )
