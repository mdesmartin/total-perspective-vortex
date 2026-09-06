"""Load a fitted pipeline and classify held-out epochs one by one."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import joblib
import numpy as np

from .preprocessing import DEFAULT_DATA_DIR, H_FREQ, L_FREQ, TMAX, TMIN, load_subject_run
from .train import _resolve_data_dir, _runs_as_list, default_model_path

# Subject: each chunk must be classified in under 2 s (no mne-realtime).
MAX_CHUNK_SECONDS = 2.0


def _resolve_model_path(
    subject: int,
    runs: list[int],
    model_path: Union[str, Path, None],
) -> Path:
    if model_path is None:
        model_path = default_model_path(subject, runs)
    else:
        model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path}\n"
            f"  Train first: python mybci.py {subject} "
            f"{' '.join(str(r) for r in runs)} train"
        )
    return model_path


def _params_from_artifact(
    artifact: object,
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
) -> dict:
    """Prefer preprocessing params stored at train time when present."""
    params = {"tmin": tmin, "tmax": tmax, "l_freq": l_freq, "h_freq": h_freq}
    if isinstance(artifact, dict):
        params.update(artifact.get("params", {}))
    return params


def _load_epochs(
    subject: int,
    runs: list[int],
    data_dir: Path,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_subject_run(
        subject, runs, data_dir=data_dir, verbose=False, **params
    )
    if X.shape[0] == 0:
        raise ValueError(
            f"No T1/T2 epochs for subject {subject} runs {runs} under {data_dir}"
        )
    return X, y


def _select_held_out_epochs(
    artifact: object,
    X: np.ndarray,
    y: np.ndarray,
    subject: int,
    runs: list[int],
) -> tuple[object, np.ndarray, np.ndarray]:
    """Return (pipeline, X, y) restricted to epochs held out during training.

    Old artifacts or subject/run mismatches fall back to streaming every epoch.
    """
    if not isinstance(artifact, dict):
        return artifact, X, y

    pipeline = artifact["pipeline"]
    test_idx = artifact.get("test_idx")
    if test_idx is None or len(test_idx) == 0:
        return pipeline, X, y

    if artifact.get("subject") != subject or list(artifact.get("runs", [])) != runs:
        return pipeline, X, y

    n_epochs = artifact.get("n_epochs")
    if n_epochs is not None and n_epochs != X.shape[0]:
        raise ValueError(
            f"Model was trained on {n_epochs} epochs but {X.shape[0]} were loaded; "
            "retrain so the held-out indices match"
        )

    test_idx = np.asarray(test_idx)
    return pipeline, X[test_idx], y[test_idx]


def _stream_classify(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify one epoch at a time; return (preds, chunk_times)."""
    n_epochs = X.shape[0]
    preds = np.empty(n_epochs, dtype=y.dtype)
    chunk_times = np.empty(n_epochs, dtype=np.float64)

    for i in range(n_epochs):
        t0 = time.perf_counter()
        preds[i] = pipeline.predict(X[i : i + 1])[0]
        chunk_times[i] = time.perf_counter() - t0

    return preds, chunk_times


def _report_stream(
    preds: np.ndarray,
    y: np.ndarray,
    chunk_times: np.ndarray,
) -> None:
    accuracy = float((preds == y).mean())
    print("epoch nb: [prediction] [truth] equal?")
    for i, (pred, truth) in enumerate(zip(preds, y)):
        print(f"epoch {i:02d}: [{pred}] [{truth}] {bool(pred == truth)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"max chunk time: {float(chunk_times.max()):.4f}s")
    n_slow = int((chunk_times >= MAX_CHUNK_SECONDS).sum())
    if n_slow:
        print(
            f"WARNING: {n_slow} epoch(s) took >= {MAX_CHUNK_SECONDS:.0f}s "
            "(subject limit is 2s per chunk)"
        )


def predict(
    subject: int,
    run: Union[int, list[int]],
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    *,
    model_path: Union[str, Path, None] = None,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved pipeline and classify each held-out epoch as a stream."""
    runs = _runs_as_list(run)
    data_dir = _resolve_data_dir(data_dir)
    model_path = _resolve_model_path(subject, runs, model_path)

    artifact = joblib.load(model_path)
    params = _params_from_artifact(artifact, tmin, tmax, l_freq, h_freq)
    X, y = _load_epochs(subject, runs, data_dir, params)
    pipeline, X, y = _select_held_out_epochs(artifact, X, y, subject, runs)

    preds, chunk_times = _stream_classify(pipeline, X, y)

    if verbose:
        _report_stream(preds, y, chunk_times)

    return preds, y, chunk_times
