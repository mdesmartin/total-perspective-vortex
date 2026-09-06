# Total Perspective Vortex

A brain–computer interface that decodes motor imagery from EEG recordings.
Given a subject's electroencephalographic signal, the program infers which
movement the person performed or imagined.

The dimensionality reduction step — **CSP (Common Spatial Patterns)** — is
implemented from scratch in `src/csp.py` as a scikit-learn transformer, so it
plugs directly into `sklearn.pipeline.Pipeline` and `cross_val_score`.

Dataset: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
(EEGMMIDB) from PhysioNet — 109 subjects, 64 channels, 160 Hz.

## Setup

Python 3.14 was used for the reported results. Create a virtualenv and install
the pinned dependencies:

```bash
make install
source .venv/bin/activate
```

Without make:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not part of this repository. Download it into the location the
code expects (about 3 GB):

```bash
wget -r -N -c -np -nH -P physionet_data https://physionet.org/files/eegmmidb/1.0.0/
```

This produces the layout `physionet_data/files/eegmmidb/1.0.0/S004/S004R14.edf`.
Any other location works via `--data-dir`.

## Usage

`mybci.py` is the only entry point. `1` is the subject, `5` the run.

```bash
python mybci.py 1 5 visualize   # raw vs filtered signal, then CSP maps
python mybci.py 1 5 train       # cross-validate, fit, save the model
python mybci.py 1 5 predict     # replay held-out epochs one by one
python mybci.py --baseline      # 6 experiments × 10 subjects (~15 s)
python mybci.py                 # same, all 109 subjects (~2 min 40 s)
```

Run `train` before `predict`. A single run has only ~15 epochs, so the printed
CV and stream accuracy jump from one call to the next. That is expected. The
stable number is the table below.

`python mybci.py --help` lists optional knobs (band-pass, epoch window, seed).

## Results

`python mybci.py` scores the six experiment types over all 109 subjects with
leave-one-run-out: each fold holds out a whole recording the model never saw.

| Experiment | Task                             | Accuracy   |
| ---------- | -------------------------------- | ---------- |
| 0          | real left vs right fist          | 0.6637     |
| 1          | imagined left vs right fist      | 0.6250     |
| 2          | real both fists vs both feet     | 0.7694     |
| 3          | imagined both fists vs both feet | 0.6762     |
| 4          | all left vs right fist           | 0.6901     |
| 5          | all both fists vs both feet      | 0.7330     |
|            | **Mean of 6 experiments**        | **0.6929** |

Every experiment clears the required 60% threshold. `--baseline` on the first
10 subjects gives 0.6913. Classification latency is ~0.3 ms per epoch, well
under the 2 s budget for a streamed chunk.

## How it works

**Preprocessing** (`src/preprocessing.py`) reads the EDF file, renames channels
to the 10-20 standard and attaches the montage. It band-passes to **8–30 Hz**,
the mu and beta rhythms where motor imagery modulates power, then cuts epochs
**0–4 s** after each cue and keeps the two task annotations (`T1`, `T2`). The
result is an array of shape `(n_epochs, 64, 641)`.

**CSP** (`src/csp.py`) finds spatial filters that maximize the variance of one
class while minimizing the other. `fit` averages the trace-normalized
covariance per class, whitens their sum, and eigendecomposes the whitened
class-1 covariance:

- `C1`, `C2` — per-class mean covariance, each epoch normalized by its trace
- `P` — whitening matrix from `C1 + C2`, so `P (C1 + C2) Pᵀ = I`
- `W = Uᵀ P` where `U` diagonalizes `P C1 Pᵀ`

The filters with the largest and smallest eigenvalues are the most
discriminative, so the top 3 and bottom 3 are kept (`n_components=6`).
`transform` projects each epoch and returns log-variance features, reducing
`64 × 641` samples to 6 numbers per epoch.

**Pipeline** (`src/pipeline.py`) chains CSP with linear discriminant analysis,
which suits the near-Gaussian log-variance features.

**Validation** happens at two levels. `src/train.py` holds out 20% of epochs
before any fitting, so `predict` streams only epochs the model never saw.
`src/evaluate.py` is stricter for the global score: it holds out a whole run,
so the spatial filters never touch the test recording.

## Layout

```
Makefile              make install — venv + pip install
mybci.py              CLI entry point
src/
  preprocessing.py    EDF loading, band-pass filter, epoching
  csp.py              CSP transformer (BaseEstimator + TransformerMixin)
  pipeline.py         sklearn Pipeline: CSP -> LDA
  train.py            cross_val_score, hold-out split, model saving
  predict.py          streamed epoch-by-epoch classification
  evaluate.py         leave-one-run-out scoring over the six experiments
  visualize.py        raw vs filtered plots, and CSP filters and features
artifacts/            generated models and figures (not versioned)
```
