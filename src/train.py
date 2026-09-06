"""Train the CSP → LDA pipeline: cross-validate, fit, and save."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .pipeline import DEFAULT_N_COMPONENTS, make_pipeline
from .preprocessing import DEFAULT_DATA_DIR, H_FREQ, L_FREQ, TMAX, TMIN, load_subject_run

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MODEL_DIR = _ROOT / "artifacts"


def _runs_as_list(run: Union[int, list[int]]) -> list[int]:
    return [run] if isinstance(run, int) else list(run)


def _resolve_data_dir(data_dir: Union[str, Path]) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = _ROOT / data_dir
    return data_dir


def default_model_path(subject: int, run: Union[int, list[int]]) -> Path:
    run_part = "_".join(f"{r:02d}" for r in _runs_as_list(run))
    return DEFAULT_MODEL_DIR / f"model_S{subject:03d}_R{run_part}.pkl"


def _load_epochs(
    subject: int,
    runs: list[int],
    data_dir: Path,
    params: dict,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_subject_run(
        subject, runs, data_dir=data_dir, verbose=verbose, **params
    )
    if X.shape[0] == 0:
        raise ValueError(
            f"No T1/T2 epochs for subject {subject} runs {runs} under {data_dir}"
        )
    return X, y


def _hold_out_test_set(
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx), stratified, test indices in chrono order."""
    idx = np.arange(y.shape[0])
    if test_size <= 0:
        return idx, np.empty(0, dtype=idx.dtype)

    n_test = int(round(y.shape[0] * test_size))
    n_classes = int(np.unique(y).size)
    if n_test < n_classes:
        raise ValueError(
            f"test_size={test_size} leaves {n_test} held-out epoch(s) for "
            f"{n_classes} classes; train on more runs or raise --test-size"
        )

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(test_idx)


def _cross_validate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
    n_components: int,
    random_state: int,
) -> tuple[object, np.ndarray]:
    """Run StratifiedKFold CV; return (unfitted pipeline, fold scores)."""
    _, class_counts = np.unique(y_train, return_counts=True)
    n_splits = min(n_splits, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 epochs per class for CV, got counts {class_counts}"
        )

    pipeline = make_pipeline(n_components=n_components)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    return pipeline, scores


def _fit_and_save(
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: Path,
    artifact: dict,
) -> None:
    pipeline.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**artifact, "pipeline": pipeline}, model_path)


def _report_scores(
    scores: np.ndarray,
    model_path: Path | None,
) -> None:
    print(np.array2string(scores, precision=4, floatmode="fixed"))
    print(f"cross_val_score: {float(scores.mean()):.4f}")
    if model_path is not None:
        print(f"Model saved in {model_path}")


def train(
    subject: int,
    run: Union[int, list[int]],
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    n_components: int = DEFAULT_N_COMPONENTS,
    test_size: float = DEFAULT_TEST_SIZE,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    model_path: Union[str, Path, None] = None,
    random_state: int | None = None,
    verbose: bool = True,
    save: bool = True,
) -> np.ndarray:
    """Cross-validate, fit, and save the pipeline for one subject."""
    runs = _runs_as_list(run)
    data_dir = _resolve_data_dir(data_dir)
    if random_state is None:
        random_state = int(np.random.default_rng().integers(2**31))
    if model_path is None:
        model_path = default_model_path(subject, runs)
    else:
        model_path = Path(model_path)

    params = {"tmin": tmin, "tmax": tmax, "l_freq": l_freq, "h_freq": h_freq}
    X, y = _load_epochs(subject, runs, data_dir, params, verbose)

    train_idx, test_idx = _hold_out_test_set(y, test_size, random_state)
    X_train, y_train = X[train_idx], y[train_idx]

    pipeline, scores = _cross_validate(
        X_train, y_train, n_splits, n_components, random_state
    )

    if save:
        _fit_and_save(
            pipeline,
            X_train,
            y_train,
            model_path,
            artifact={
                "subject": subject,
                "runs": runs,
                "test_idx": test_idx,
                "n_epochs": int(X.shape[0]),
                "params": params,
                "seed": random_state,
            },
        )

    if verbose:
        _report_scores(scores, model_path if save else None)

    return scores
