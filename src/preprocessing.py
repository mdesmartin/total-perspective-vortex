"""Load, band-pass and epoch EEG from PhysioNet EEGMMIDB."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import mne
import numpy as np
from mne.datasets import eegbci

mne.set_log_level("WARNING")

# Known EEGMMIDB EDF quirks (S100 all runs, S067R13). Filter by message so
# other RuntimeWarnings stay visible.
_KNOWN_EDF_WARNINGS = (
    "Limited .* annotation",
    "Number of records from the ?header does not match the file size",
)

DEFAULT_DATA_DIR = Path("physionet_data/files/eegmmidb/1.0.0")

L_FREQ = 8.0    # Hz — mu+beta low edge
H_FREQ = 30.0   # Hz — mu+beta high edge
TMIN = 0.0      # s relative to cue onset
TMAX = 4.0      # s relative to cue onset

TASK_LABELS = ("T1", "T2")


def _edf_path(data_dir: Path, subject: int, run: int) -> Path:
    return data_dir / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def load_raw(data_dir: Path, subject: int, run: int) -> mne.io.BaseRaw:
    """Read one EDF, standardize channel names, set the 10-20 montage."""
    path = _edf_path(data_dir, subject, run)
    if not path.exists():
        raise FileNotFoundError(
            f"EDF file not found: {path}\n"
            f"  Expected layout: <data_dir>/S{{subject:03d}}/S{{subject:03d}}R{{run:02d}}.edf"
        )
    with warnings.catch_warnings():
        for message in _KNOWN_EDF_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)
        raw = mne.io.read_raw_edf(str(path), preload=True)
    eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    return raw


def bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
) -> mne.io.BaseRaw:
    """Band-pass ``raw`` in place (mu+beta by default) and return it."""
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        l_trans_bandwidth=2.0,
        h_trans_bandwidth=2.0,
        fir_window="hamming",
    )
    return raw


def _epoch_t1_t2(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for T1/T2 only; empty arrays if the run has no task events."""
    events, event_id = mne.events_from_annotations(raw)
    task_event_id = {k: v for k, v in event_id.items() if k in TASK_LABELS}

    if not task_event_id:
        n_ch = len(raw.ch_names)
        n_t = int((tmax - tmin) * raw.info["sfreq"]) + 1
        return np.empty((0, n_ch, n_t), dtype=np.float64), np.empty(0, dtype=np.int64)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=task_event_id,
        tmin=tmin,
        tmax=tmax,
        picks="eeg",
        baseline=None,
        preload=True,
    )
    X = epochs.get_data().astype(np.float64)
    y = epochs.events[:, 2].astype(np.int64)
    return X, y


def _load_and_concat_raws(
    subject: int,
    runs: list[int],
    data_dir: Path,
) -> mne.io.BaseRaw:
    raws = [load_raw(data_dir, subject, r) for r in runs]
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws)


def _print_summary(
    subject: int,
    runs: list[int],
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    print(f"Subject S{subject:03d}  runs {runs}")
    print(f"  X : {X.shape}  (epochs × channels × times)")
    print(f"  y : {y.shape}  labels {np.unique(y).tolist()}")
    if y.size > 0:
        unique, counts = np.unique(y, return_counts=True)
        for code, count in zip(unique, counts):
            print(f"    class {code}: {count} epochs")


def load_subject_run(
    subject: int,
    run: Union[int, list[int]],
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    *,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one or several runs → band-pass → epoch T1/T2 → (X, y)."""
    data_dir = Path(data_dir)
    runs = [run] if isinstance(run, int) else list(run)

    raw = _load_and_concat_raws(subject, runs, data_dir)
    bandpass(raw, l_freq, h_freq)
    X, y = _epoch_t1_t2(raw, tmin, tmax)

    if verbose:
        _print_summary(subject, runs, X, y)

    return X, y
