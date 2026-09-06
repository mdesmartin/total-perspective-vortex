"""
mybci.py — CLI entry point for Total Perspective Vortex.

Usage
-----
python mybci.py 1 5 visualize   # raw vs filtered signal, then CSP maps
python mybci.py 1 5 train       # cross-validate, fit, save the model
python mybci.py 1 5 predict     # replay held-out epochs one by one
python mybci.py --baseline      # 6 experiments × 10 subjects
python mybci.py                 # same, all 109 subjects

Run ``python mybci.py --help`` for optional knobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODES = ("visualize", "train", "predict")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.evaluate import EXPERIMENTS
    from src.pipeline import DEFAULT_N_COMPONENTS
    from src.preprocessing import DEFAULT_DATA_DIR, H_FREQ, L_FREQ, TMAX, TMIN
    from src.train import DEFAULT_N_SPLITS, DEFAULT_TEST_SIZE

    parser = argparse.ArgumentParser(
        description="Visualize, train or predict on EEGMMIDB motor imagery data.",
        usage="python mybci.py [<subject> <run>... <%s>]" % "|".join(MODES),
    )
    parser.add_argument(
        "args",
        nargs="*",
        metavar="ARG",
        help=f"<subject> <run>... <{'|'.join(MODES)}>. Omit for the full evaluation.",
    )

    data = parser.add_argument_group("data and preprocessing")
    data.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    data.add_argument("--tmin", type=float, default=TMIN)
    data.add_argument("--tmax", type=float, default=TMAX)
    data.add_argument("--l-freq", type=float, default=L_FREQ)
    data.add_argument("--h-freq", type=float, default=H_FREQ)

    model = parser.add_argument_group("model and validation")
    model.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    model.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    model.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    model.add_argument("--seed", type=int, default=None)
    model.add_argument("-m", "--model", type=Path, default=None)

    figures = parser.add_argument_group("visualize")
    figures.add_argument("-o", "--output", type=Path, default=None)
    figures.add_argument("--no-show", action="store_true")
    figures.add_argument("--no-csp", action="store_true")

    evaluation = parser.add_argument_group("full evaluation")
    evaluation.add_argument("--baseline", action="store_true")
    evaluation.add_argument("--n-subjects", type=int, default=None)
    evaluation.add_argument(
        "--experiment",
        type=int,
        default=None,
        choices=[e[0] for e in EXPERIMENTS],
    )
    return parser.parse_args(argv)


def _split_positionals(args: list[str]) -> tuple[int, list[int], str]:
    """Split ``["1", "5", "9", "13", "train"]`` into (1, [5, 9, 13], "train")."""
    if len(args) < 3 or args[-1] not in MODES:
        raise ValueError(
            f"usage: python mybci.py <subject> <run>... {'|'.join(MODES)}"
        )
    mode = args[-1]
    try:
        numbers = [int(a) for a in args[:-1]]
    except ValueError:
        raise ValueError(f"subject and runs must be integers, got {args[:-1]}") from None

    subject, runs = numbers[0], numbers[1:]
    if not 1 <= subject <= 109:
        raise ValueError(f"subject {subject} out of range 1..109")
    for run in runs:
        if not 1 <= run <= 14:
            raise ValueError(f"run {run} out of range 1..14")
    return subject, runs, mode


def _preprocess_kwargs(args: argparse.Namespace) -> dict:
    return {
        "data_dir": args.data_dir,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "l_freq": args.l_freq,
        "h_freq": args.h_freq,
    }


def _run_evaluation(args: argparse.Namespace) -> None:
    from src.evaluate import (
        DEFAULT_N_SUBJECTS,
        EXPERIMENTS,
        N_ALL_SUBJECTS,
        run_experiments,
    )

    n_subjects = args.n_subjects
    if n_subjects is None:
        n_subjects = DEFAULT_N_SUBJECTS if args.baseline else N_ALL_SUBJECTS
    experiments = EXPERIMENTS
    if args.experiment is not None:
        experiments = tuple(e for e in EXPERIMENTS if e[0] == args.experiment)

    run_experiments(
        n_subjects=n_subjects,
        n_components=args.n_components,
        experiments=experiments,
        **_preprocess_kwargs(args),
    )


def _run_visualize(
    args: argparse.Namespace, subject: int, run, runs: list[int]
) -> None:
    from src.visualize import DEFAULT_FIGURE_DIR, visualize

    output = args.output
    if args.no_show and output is None:
        run_part = "_".join(f"{r:02d}" for r in runs)
        output = DEFAULT_FIGURE_DIR / f"preprocessing_S{subject:03d}_R{run_part}.png"
    visualize(
        subject=subject,
        run=run,
        save_path=output,
        show=not args.no_show,
        include_csp=not args.no_csp,
        **_preprocess_kwargs(args),
    )


def _run_train(args: argparse.Namespace, subject: int, run, _runs: list[int]) -> None:
    from src.train import train

    train(
        subject=subject,
        run=run,
        n_splits=args.n_splits,
        n_components=args.n_components,
        test_size=args.test_size,
        model_path=args.model,
        random_state=args.seed,
        **_preprocess_kwargs(args),
    )


def _run_predict(args: argparse.Namespace, subject: int, run, _runs: list[int]) -> None:
    from src.predict import predict

    predict(
        subject=subject,
        run=run,
        model_path=args.model,
        **_preprocess_kwargs(args),
    )


_HANDLERS = {
    "visualize": _run_visualize,
    "train": _run_train,
    "predict": _run_predict,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.args:
        _run_evaluation(args)
        return 0

    try:
        subject, runs, mode = _split_positionals(args.args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run = runs[0] if len(runs) == 1 else runs
    _HANDLERS[mode](args, subject, run, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
