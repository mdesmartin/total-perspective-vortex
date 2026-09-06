"""Visualize raw vs filtered EEG, and CSP spatial filters with their features."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import mne
import numpy as np

from .csp import CSP
from .pipeline import DEFAULT_N_COMPONENTS
from .preprocessing import (
    DEFAULT_DATA_DIR, H_FREQ, L_FREQ, TMAX, TMIN,
    bandpass, load_raw, load_subject_run,
)

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHANNELS = ("C3", "Cz", "C4")
DEFAULT_DURATION = 10.0
PSD_FMAX = 60.0
DEFAULT_FIGURE_DIR = _ROOT / "artifacts"
TASK_NAMES = ("T1", "T2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    return data_dir if data_dir.is_absolute() else _ROOT / data_dir


def _pick_channels(raw, requested: tuple[str, ...]) -> list[str]:
    available = [ch for ch in requested if ch in raw.ch_names]
    return available or raw.ch_names[: len(requested)]


def _class_labels(classes: np.ndarray) -> list[str]:
    return [
        f"{TASK_NAMES[i]} (code {int(c)})" if i < len(TASK_NAMES) else f"code {int(c)}"
        for i, c in enumerate(classes)
    ]


def _save_figure(fig, save_path: str | Path | None) -> Path | None:
    if save_path is None:
        return None
    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = _ROOT / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Figure saved in {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------

def _plot_time_series(ax, raw, channels: list[str], duration: float, title: str) -> None:
    sfreq = raw.info["sfreq"]
    n_times = min(int(duration * sfreq), raw.n_times)
    data_uv = raw.get_data(picks=channels, start=0, stop=n_times) * 1e6
    times = np.arange(n_times) / sfreq

    offset = 3.0 * np.median(np.abs(data_uv - data_uv.mean(axis=1, keepdims=True))) or 1.0
    for i, (ch, trace) in enumerate(zip(channels, data_uv)):
        ax.plot(times, trace - trace.mean() + i * offset, linewidth=0.7, label=ch)

    ax.set_yticks([i * offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    ax.set_xlabel("time (s)")
    ax.set_xlim(times[0], times[-1])
    ax.set_title(title)


def _plot_psd(ax, raw, channels: list[str], title: str, band: tuple[float, float]) -> None:
    psd = raw.compute_psd(picks=channels, fmax=PSD_FMAX)
    power_db = 10.0 * np.log10(np.maximum(psd.get_data(), 1e-30))

    for ch, curve in zip(channels, power_db):
        ax.plot(psd.freqs, curve, linewidth=0.9, label=ch)

    ax.axvspan(band[0], band[1], color="tab:green", alpha=0.12, label="kept band")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB)")
    ax.set_xlim(0, PSD_FMAX)
    ax.set_title(title)
    ax.legend(fontsize="x-small", loc="upper right")


def _plot_csp_topographies(axes, csp: CSP, info) -> None:
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(csp.filters_[i], info, axes=ax, show=False)
        ax.set_title(f"w{i + 1}\nλ={csp.eigenvalues_[i]:.2f}", fontsize="small")


def _plot_component_variance(ax, features, y, classes, labels) -> None:
    idx = np.arange(features.shape[1])
    width = 0.38
    for k, (code, label) in enumerate(zip(classes, labels)):
        subset = features[y == code]
        ax.bar(
            idx + (k - 0.5) * width,
            subset.mean(axis=0), width,
            yerr=subset.std(axis=0), capsize=2, alpha=0.85, label=label,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"w{i + 1}" for i in idx])
    ax.set_ylabel("log-variance")
    ax.set_title("CSP features by class (mean ± sd)")
    ax.legend(fontsize="x-small")


def _plot_feature_scatter(ax, features, y, classes, labels) -> None:
    n = features.shape[1]
    for code, label in zip(classes, labels):
        s = features[y == code]
        ax.scatter(s[:, 0], s[:, -1], s=30, alpha=0.8, label=label)
    ax.set_xlabel("first component (w1)")
    ax.set_ylabel(f"last component (w{n})")
    ax.set_title("Class separation in the extreme components")
    ax.legend(fontsize="x-small")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def csp_figure(
    subject: int,
    run: Union[int, list[int]],
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Fit CSP on the given epochs and plot topographies + feature charts.

    Returns None if the runs have no task epochs or only one class.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = _resolve_data_dir(data_dir)
    runs = [run] if isinstance(run, int) else list(run)

    X, y = load_subject_run(
        subject, runs, data_dir=data_dir,
        tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq, verbose=False,
    )
    if X.shape[0] == 0 or np.unique(y).size < 2:
        print(f"  skipping CSP figure: {X.shape[0]} epoch(s), {np.unique(y).size} class(es)")
        return None

    csp = CSP(n_components=n_components).fit(X, y)
    features = csp.transform(X)
    labels = _class_labels(csp.classes_)

    raw = load_raw(data_dir, subject, runs[0])
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=False, eeg=True))

    print(f"  CSP: {X.shape[0]} epochs, {n_components} filters from {X.shape[1]} channels")
    print(f"  eigenvalues: {np.array2string(csp.eigenvalues_, precision=3, floatmode='fixed')}")

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, n_components, height_ratios=[1.0, 1.4], hspace=0.45)
    topo_axes = [fig.add_subplot(grid[0, i]) for i in range(n_components)]
    ax_var = fig.add_subplot(grid[1, : n_components // 2])
    ax_scatter = fig.add_subplot(grid[1, n_components // 2 :])

    _plot_csp_topographies(topo_axes, csp, info)
    _plot_component_variance(ax_var, features, y, csp.classes_, labels)
    _plot_feature_scatter(ax_scatter, features, y, csp.classes_, labels)

    run_part = ", ".join(f"R{r:02d}" for r in runs)
    fig.suptitle(f"S{subject:03d} {run_part} — CSP ({X.shape[0]} epochs, in-sample)")

    _save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize(
    subject: int,
    run: Union[int, list[int]],
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    tmin: float = TMIN,
    tmax: float = TMAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    duration: float = DEFAULT_DURATION,
    save_path: str | Path | None = None,
    show: bool = True,
    include_csp: bool = True,
):
    """Preprocessing figure (raw vs filtered) and optionally a CSP figure."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = _resolve_data_dir(data_dir)
    runs = [run] if isinstance(run, int) else list(run)

    raw_before = load_raw(data_dir, subject, runs[0])
    raw_after = bandpass(raw_before.copy(), l_freq, h_freq)
    picks = _pick_channels(raw_before, channels)

    print(f"Subject S{subject:03d}  run R{runs[0]:02d}")
    print(f"  {len(raw_before.ch_names)} channels @ {raw_before.info['sfreq']:.0f} Hz")
    print(f"  duration: {raw_before.times[-1]:.1f}s")
    print(f"  plotted channels: {', '.join(picks)}")
    print(f"  band-pass: {l_freq:g}-{h_freq:g} Hz")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _plot_time_series(axes[0, 0], raw_before, picks, duration, "Raw signal (unfiltered)")
    _plot_psd(axes[0, 1], raw_before, picks, "Raw spectrum", (l_freq, h_freq))
    _plot_time_series(axes[1, 0], raw_after, picks, duration, f"Filtered {l_freq:g}-{h_freq:g} Hz")
    _plot_psd(axes[1, 1], raw_after, picks, "Filtered spectrum", (l_freq, h_freq))
    fig.suptitle(f"S{subject:03d} R{runs[0]:02d} — raw vs {l_freq:g}-{h_freq:g} Hz band-pass")
    fig.tight_layout()

    csp_save_path = None
    saved = _save_figure(fig, save_path)
    if saved is not None:
        csp_save_path = saved.with_name(f"{saved.stem}_csp{saved.suffix}")

    figures = [fig]
    if show:
        plt.show()
    else:
        plt.close(fig)

    if include_csp:
        csp_fig = csp_figure(
            subject, runs, data_dir,
            tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq,
            save_path=csp_save_path, show=show,
        )
        if csp_fig is not None:
            figures.append(csp_fig)

    return figures
