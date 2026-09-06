"""Leave-one-run-out accuracy across the six motor-imagery experiment families."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .pipeline import DEFAULT_N_COMPONENTS, make_pipeline
from .preprocessing import DEFAULT_DATA_DIR, H_FREQ, L_FREQ, TMAX, TMIN, load_subject_run
from .train import _resolve_data_dir

MOTOR_EXPERIMENTS: tuple[tuple[int, tuple[int, ...], str], ...] = (
    (0, (3, 7, 11), "real left vs right fist"),
    (1, (4, 8, 12), "imagery left vs right fist"),
    (2, (5, 9, 13), "real both fists vs both feet"),
    (3, (6, 10, 14), "imagery both fists vs both feet"),
)

# Same four families plus real+imagery concatenated (more train epochs).
EXPERIMENTS: tuple[tuple[int, tuple[int, ...], str], ...] = MOTOR_EXPERIMENTS + (
    (4, (3, 4, 7, 8, 11, 12), "all left vs right fist"),
    (5, (5, 6, 9, 10, 13, 14), "all both fists vs both feet"),
)

DEFAULT_N_SUBJECTS = 10
N_ALL_SUBJECTS = 109


def _load_experiment_runs(
    subject: int,
    runs: list[int],
    data_dir: Path,
    params: dict,
) -> list[tuple[np.ndarray, np.ndarray]]:
    loaded: list[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        X, y = load_subject_run(
            subject, run, data_dir=data_dir, verbose=False, **params
        )
        if X.shape[0] == 0:
            raise ValueError(f"S{subject:03d} run {run}: no T1/T2 epochs")
        loaded.append((X, y))
    return loaded


def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> float:
    pipeline = make_pipeline(n_components=n_components)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return float((preds == y_test).mean())


def _leave_one_run_out(
    subject: int,
    runs: list[int],
    loaded: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
) -> float:
    """Train on all but one run; average accuracy over each held-out run."""
    accs: list[float] = []
    for i in range(len(loaded)):
        X_train = np.concatenate([loaded[j][0] for j in range(len(loaded)) if j != i])
        y_train = np.concatenate([loaded[j][1] for j in range(len(loaded)) if j != i])
        X_test, y_test = loaded[i]
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"S{subject:03d}: need two classes when holding out run {runs[i]}"
            )
        accs.append(_fit_and_score(X_train, y_train, X_test, y_test, n_components))
    return float(np.mean(accs))


def experiment_score(
    subject: int,
    runs: Sequence[int],
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
) -> float:
    """Leave-one-run-out accuracy for one subject on one experiment family."""
    data_dir = _resolve_data_dir(data_dir)
    runs = list(runs)
    params = {"tmin": tmin, "tmax": tmax, "l_freq": l_freq, "h_freq": h_freq}
    loaded = _load_experiment_runs(subject, runs, data_dir, params)
    return _leave_one_run_out(subject, runs, loaded, n_components)


def _score_subjects(
    exp_id: int,
    runs: tuple[int, ...],
    n_subjects: int,
    data_dir: Path,
    n_components: int,
    params: dict,
    verbose: bool,
) -> np.ndarray:
    accs: list[float] = []
    for subject in range(1, n_subjects + 1):
        try:
            acc = experiment_score(
                subject,
                runs,
                data_dir=data_dir,
                n_components=n_components,
                **params,
            )
        except Exception as exc:
            if verbose:
                print(
                    f"experiment {exp_id}: subject {subject:03d}: SKIP ({exc})",
                    flush=True,
                )
            continue
        accs.append(acc)
        if verbose:
            print(
                f"experiment {exp_id}: subject {subject:03d}: "
                f"accuracy = {acc:.4f}",
                flush=True,
            )
    return np.asarray(accs, dtype=np.float64)


def _report_experiment_mean(exp_id: int, accs: np.ndarray) -> None:
    if accs.size:
        print(f"experiment {exp_id}: mean = {float(accs.mean()):.4f}  (n={len(accs)})")
        print()


def _report_overall_means(
    results: dict[int, np.ndarray],
    n_subjects: int,
) -> None:
    means = [float(v.mean()) for v in results.values() if v.size]
    if not means:
        return
    who = (
        f"all {n_subjects} subjects"
        if n_subjects >= N_ALL_SUBJECTS
        else f"{n_subjects} subjects"
    )
    print(f"Mean accuracy of the six different experiments for {who}:")
    for exp_id, accs in results.items():
        if accs.size:
            print(f"experiment {exp_id}: accuracy = {float(accs.mean()):.4f}")
    print(f"Mean accuracy of {len(means)} experiments: {float(np.mean(means)):.4f}")


def run_experiments(
    n_subjects: int = DEFAULT_N_SUBJECTS,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    experiments: Sequence[tuple[int, tuple[int, ...], str]] = EXPERIMENTS,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """Score the first ``n_subjects`` with leave-one-run-out on each experiment."""
    data_dir = _resolve_data_dir(data_dir)
    params = {"tmin": tmin, "tmax": tmax, "l_freq": l_freq, "h_freq": h_freq}
    results: dict[int, np.ndarray] = {}

    for exp_id, runs, label in experiments:
        if verbose:
            print(
                f"experiment {exp_id}: {label}  "
                f"(leave-one-run-out on {list(runs)})",
                flush=True,
            )
        accs = _score_subjects(
            exp_id, runs, n_subjects, data_dir, n_components, params, verbose
        )
        results[exp_id] = accs
        if verbose:
            _report_experiment_mean(exp_id, accs)

    if verbose and results:
        _report_overall_means(results, n_subjects)

    return results
