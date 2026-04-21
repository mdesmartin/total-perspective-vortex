"""
preprocessing.py — Load, filter and epoch EEG data from PhysioNet EEGMMIDB.

Public API
----------
load_subject_run(subject, run, data_dir, tmin, tmax, l_freq, h_freq)
    Load one or several runs for a subject and return (X, y) arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import mne
import numpy as np
from mne.datasets import eegbci

mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Constants — matches the physionet layout used in the exploration notebooks
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("physionet_data/files/eegmmidb/1.0.0")

L_FREQ = 8.0    # Hz — low edge of mu+beta band
H_FREQ = 30.0   # Hz — high edge of mu+beta band
TMIN   = 0.0    # s relative to cue onset
TMAX   = 4.0    # s relative to cue onset (cues last ~4.1 s in this dataset)

# Task labels we care about: T1 = left fist / both fists, T2 = right fist / both feet
TASK_LABELS = ("T1", "T2")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_path(data_dir: Path, subject: int, run: int) -> Path:
    return data_dir / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def _load_raw(data_dir: Path, subject: int, run: int) -> mne.io.BaseRaw:
    path = _build_path(data_dir, subject, run)
    if not path.exists():
        raise FileNotFoundError(
            f"EDF file not found: {path}\n"
            f"  Expected layout: <data_dir>/S{{subject:03d}}/S{{subject:03d}}R{{run:02d}}.edf"
        )
    raw = mne.io.read_raw_edf(str(path), preload=True)
    # Rename channels to 10-20 standard (e.g. 'Fc5.' → 'FC5')
    eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    return raw


def _filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float) -> mne.io.BaseRaw:
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        l_trans_bandwidth=2.0,
        h_trans_bandwidth=2.0,
        fir_window="hamming",
    )
    return raw


def _epoch(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a single Raw object, filtering to T1/T2 only.

    Returns empty arrays (shape (0, n_ch, n_t) and (0,)) if no task events
    are found — e.g. baseline-only runs.
    """
    events, event_id = mne.events_from_annotations(raw)

    task_event_id = {k: v for k, v in event_id.items() if k in TASK_LABELS}

    if not task_event_id:
        n_ch = len(raw.ch_names)
        n_t  = int((tmax - tmin) * raw.info["sfreq"]) + 1
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

    X = epochs.get_data().astype(np.float64)   # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2].astype(np.int64)   # integer class labels
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Load EEG data for one subject and one or several runs.

    Parameters
    ----------
    subject : int
        Subject number (1–109).
    run : int or list[int]
        Run number(s). Multiple runs are concatenated before epoching so that
        the label IDs remain consistent across runs.
    data_dir : str or Path
        Root of the EEGMMIDB local copy.
        Expected layout: <data_dir>/S001/S001R04.edf
    tmin, tmax : float
        Epoch window in seconds relative to the cue onset.
    l_freq, h_freq : float
        Bandpass filter edges in Hz.
    verbose : bool
        Print shape and label-count summary when True.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    y : np.ndarray, shape (n_epochs,)  — integer labels (T1 or T2 event codes)
    """
    data_dir = Path(data_dir)
    runs = [run] if isinstance(run, int) else list(run)

    raws = []
    for r in runs:
        try:
            raws.append(_load_raw(data_dir, subject, r))
        except FileNotFoundError as exc:
            # Surface a clear error — don't silently skip
            raise exc

    if len(raws) == 1:
        raw = raws[0]
    else:
        raw = mne.concatenate_raws(raws)

    _filter(raw, l_freq, h_freq)
    X, y = _epoch(raw, tmin, tmax)

    if verbose:
        _print_summary(subject, runs, X, y)

    return X, y


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
