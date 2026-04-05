"""CLI, evaluation plots/reports, and predictor registry. Core numerics live in ``main``."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from SES import Predictor

HOME = Path(os.environ.get("STOCK_DATA_HOME", str(Path(__file__).resolve().parent)))
OUTPUT_DIR = HOME / "output"
SAMPLE_DATA_DIR = HOME / "sample_data"

PREDICTOR_DISCOVERY_MODULES: tuple[str, ...] = ("SES",)

_plot_seq = 0


def normalize_sample_name(name: str) -> str:
    name = name.strip()
    return name[:-4] if name.lower().endswith(".npy") else name


def sample_npy_path(stem: str) -> Path:
    base = normalize_sample_name(stem)
    return SAMPLE_DATA_DIR / f"{base}.npy"


def resolve_pv_path(stk: str) -> Path:
    if os.environ.get("STOCK_PV_NPY"):
        return Path(os.environ["STOCK_PV_NPY"])
    p = sample_npy_path(stk)
    if not p.is_file():
        raise FileNotFoundError(f"Missing sample data file: {p}")
    return p


def list_sample_npy_files() -> list[Path]:
    if not SAMPLE_DATA_DIR.is_dir():
        return []
    return sorted(SAMPLE_DATA_DIR.glob("*.npy"), key=lambda p: p.name.lower())


def save_figure(fig: matplotlib.figure.Figure, stem: str) -> Path:
    global _plot_seq
    _plot_seq += 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{_plot_seq:02d}_{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path.resolve()}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path


def save_figure_path(fig: matplotlib.figure.Figure, path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / path.name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path.resolve()}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path


def plot_eval_figure(
    stk: str, e: np.ndarray, base_m: float, abs_m: float, rel_m: float
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    fig.suptitle(f"Prediction evaluation — {stk}", fontsize=13, fontweight="bold")
    ax.hist(e, bins=30, edgecolor="black", alpha=0.85, label="Error: t[i] - p[i-1]")
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    metrics_lines = f"base = {base_m}\nabs = {abs_m}\nrel = {rel_m}"
    ax.text(
        0.98,
        0.98,
        metrics_lines,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.92},
    )
    return fig


def _coerce_param_value(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s, 10)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_algo_params(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``['alpha=0.3', 'beta=1']`` into a dict for predictor ``__init__``."""
    if not pairs:
        return {}
    out: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid --param {raw!r}; expected KEY=VALUE")
        key, _, rest = raw.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --param {raw!r}; missing key")
        out[key] = _coerce_param_value(rest)
    return out


def _init_kwargs_for_predictor(cls: type, params: Mapping[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    accepted: set[str] = set()
    for name, par in sig.parameters.items():
        if name == "self":
            continue
        if par.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        accepted.add(name)
    return {k: v for k, v in params.items() if k in accepted}


class DefaultPredictor(Predictor):
    """Assignment template (truncated MA); delegates to ``main.prediction_default``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} does not accept init parameters: {sorted(kwargs)}"
            )

    def predict(self, P, V, h=20):
        import main as m

        return m.prediction_default(P, V, h)


def _predictor_classes_in_module(module: types.ModuleType) -> list[type[Predictor]]:
    found: list[type[Predictor]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Predictor or not issubclass(obj, Predictor):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        found.append(obj)
    return sorted(found, key=lambda c: c.__name__)


def _predictor_registry() -> dict[str, type[Predictor]]:
    reg: dict[str, type[Predictor]] = {
        "default": DefaultPredictor,
        DefaultPredictor.__name__.lower(): DefaultPredictor,
    }
    for mod_name in PREDICTOR_DISCOVERY_MODULES:
        mod = importlib.import_module(mod_name)
        for cls in _predictor_classes_in_module(mod):
            key = cls.__name__.lower()
            if key in reg and reg[key] is not cls:
                raise ValueError(
                    f"Predictor name collision for {key!r}: {reg[key].__name__} vs {cls.__name__}"
                )
            reg[key] = cls
    return reg


def resolve_predictor_class(name: str) -> type[Predictor]:
    key = name.strip().lower()
    reg = _predictor_registry()
    if key not in reg:
        known = ", ".join(registered_predictor_names())
        raise ValueError(f"Unknown predictor {name!r}. Known: {known}")
    cls = reg[key]
    if not issubclass(cls, Predictor):
        raise TypeError(f"{cls.__name__} is not a Predictor subclass")
    return cls


def registered_predictor_names() -> tuple[str, ...]:
    reg = _predictor_registry()
    names = {cls.__name__ for cls in set(reg.values())}
    names.add("default")
    return tuple(sorted(names, key=str.lower))


def make_predictor(
    algo: str,
    params: Mapping[str, Any] | None = None,
) -> tuple[str, Callable[..., np.ndarray]]:
    params = dict(params or {})
    cls = resolve_predictor_class(algo)
    kw = _init_kwargs_for_predictor(cls, params)
    try:
        model = cls(**kw)
    except TypeError as ex:
        raise ValueError(f"Invalid parameters for {cls.__name__}: {ex}") from ex

    def predict_bound(P, V, h=20):
        return model.predict(P, V, h)

    if kw:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(kw.items()))
        label = f"{cls.__name__}({inner})"
    else:
        label = cls.__name__
    return label, predict_bound


def run_evaluation(
    stk: str,
    *,
    predict: Callable[..., np.ndarray] | None = None,
    h: int = 5,
    algo_label: str | None = None,
) -> None:
    import main as core

    if algo_label:
        print(f"Predictor: {algo_label}")
    P, V = core.load_data(stk)
    e, num, den, rel = core.compute_eval_from_pv(P, V, predict=predict, h=h)
    base_m = round(num, 3)
    abs_m = round(den, 3)
    rel_m = round(rel, 3)
    print(f"\n\tbase = {base_m}  |  abs = {abs_m}  |  rel = {rel_m}\n")

    fig = plot_eval_figure(stk, e, base_m, abs_m, rel_m)
    save_figure(fig, "prediction_eval")


def run_evaluation_all(
    no_plot: bool = False,
    *,
    predict: Callable[..., np.ndarray] | None = None,
    h: int = 5,
    algo_label: str | None = None,
) -> None:
    import main as core

    paths = list_sample_npy_files()
    if not paths:
        raise SystemExit(f"No .npy files in {SAMPLE_DATA_DIR}")

    rows: list[tuple[str, float, float, float, float, float, int]] = []
    all_e: list[np.ndarray] = []

    for path in paths:
        stem = path.stem
        try:
            P, V = core.load_data_path(path)
            e, num, den, rel = core.compute_eval_from_pv(P, V, predict=predict, h=h)
        except Exception as ex:
            print(f"[skip] {stem}: {ex}")
            continue

        base_m = round(num, 3)
        abs_m = round(den, 3)
        rel_m = round(rel, 3)
        err_min = float(np.nanmin(e))
        err_max = float(np.nanmax(e))
        n_e = int(e.size)
        rows.append((stem, num, den, rel, err_min, err_max, n_e))
        all_e.append(e)

        print(f"{stem}: base={base_m} abs={abs_m} rel={rel_m}  err_min={err_min:.6g} err_max={err_max:.6g}")

        if not no_plot:
            fig = plot_eval_figure(stem, e, base_m, abs_m, rel_m)
            save_figure_path(fig, Path(f"prediction_eval_{stem}.png"))

    if not rows:
        raise SystemExit("No sample files evaluated successfully.")

    bases = np.array([r[1] for r in rows], dtype=float)
    abses = np.array([r[2] for r in rows], dtype=float)
    rels = np.array([r[3] for r in rows], dtype=float)

    e_pool = np.concatenate(all_e) if all_e else np.array([])
    pool_min = float(np.nanmin(e_pool)) if e_pool.size else float("nan")
    pool_max = float(np.nanmax(e_pool)) if e_pool.size else float("nan")
    pool_mean = float(np.nanmean(e_pool)) if e_pool.size else float("nan")
    pool_std = float(np.nanstd(e_pool, ddof=1)) if e_pool.size > 1 else 0.0

    def fmt_mean_std(x: np.ndarray) -> str:
        m, s = float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return f"{m:.6g} +/- {s:.6g}"

    lines: list[str] = []
    lines.append("Evaluation report - all sample_data/*.npy")
    lines.append(f"Predictor: {algo_label or 'default'}  |  h = {h}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"{'file':<12} {'base':>10} {'abs':>10} {'rel':>10} {'err_min':>14} {'err_max':>14} {'n':>6}"
    )
    lines.append("-" * 60)
    for r in rows:
        stem, num, den, rel, emi, ema, n_e = r
        lines.append(
            f"{stem:<12} {num:10.6g} {den:10.6g} {rel:10.6g} {emi:14.6g} {ema:14.6g} {n_e:6d}"
        )
    lines.append("")
    lines.append("Summary across files (one row per .npy)")
    lines.append("-" * 60)
    lines.append(f"  base: {fmt_mean_std(bases)}    [min {np.min(bases):.6g}, max {np.max(bases):.6g}]")
    lines.append(f"  abs:  {fmt_mean_std(abses)}    [min {np.min(abses):.6g}, max {np.max(abses):.6g}]")
    lines.append(f"  rel:  {fmt_mean_std(rels)}    [min {np.min(rels):.6g}, max {np.max(rels):.6g}]")
    lines.append("")
    lines.append("Pooled prediction errors (all files concatenated)")
    lines.append("-" * 60)
    lines.append(f"  count: {e_pool.size}")
    lines.append(f"  error min: {pool_min:.6g}")
    lines.append(f"  error max: {pool_max:.6g}")
    lines.append(f"  error mean +/- std: {pool_mean:.6g} +/- {pool_std:.6g}")
    lines.append("")
    report_text = "\n".join(lines)

    print()
    print(report_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote report: {report_path.resolve()}")


def cli_source_name(stock: str | None, sample: str | None, source: str | None) -> str | None:
    if stock and sample:
        raise SystemExit("Use only one of --stock and --sample.")
    if source and (stock or sample):
        raise SystemExit("Do not combine positional SOURCE with --stock/--sample.")

    if stock:
        return normalize_sample_name(stock)
    if sample:
        return normalize_sample_name(sample)
    if source:
        return normalize_sample_name(source)
    return None


def ensure_sample_exists(stem: str) -> None:
    p = sample_npy_path(stem)
    if not p.is_file():
        raise SystemExit(f"Missing sample data file: {p}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run prediction + evaluation on sample_data/*.npy."
    )
    p.add_argument(
        "source",
        nargs="?",
        metavar="SOURCE",
        help="Stem of sample_data/*.npy (e.g. s1). Ignored if --all.",
    )
    p.add_argument("--stock", metavar="SYM", help="Same as SOURCE.")
    p.add_argument("--sample", metavar="NAME", help="Same as SOURCE (.npy optional).")
    p.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every sample_data/*.npy and write evaluation_report.txt.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="With --all: skip per-file PNG charts (console + evaluation_report.txt only).",
    )
    p.add_argument(
        "--algo",
        default="default",
        metavar="NAME",
        help=(
            "Special alias ``default`` (assignment template) or full class name of a "
            f"discovered Predictor (e.g. SimpleExponentialSmoothing). Known: {', '.join(registered_predictor_names())}."
        ),
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Keyword argument for the selected predictor's __init__ (repeatable). "
            "Example: --param alpha=0.3"
        ),
    )
    p.add_argument(
        "--predict-h",
        type=int,
        default=5,
        metavar="H",
        help="Value of h passed to predict(P, V, h) (MA window for default; SES ignores it).",
    )
    return p.parse_args(argv)


def main_cli(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.no_plot and not args.all:
        raise SystemExit("--no-plot only applies with --all.")

    try:
        params = parse_algo_params(list(args.param or []))
        algo_label, predict_fn = make_predictor(args.algo.strip(), params)
    except ValueError as ex:
        raise SystemExit(str(ex)) from ex

    ph = args.predict_h
    if ph < 1:
        raise SystemExit("--predict-h must be >= 1.")

    if args.all:
        if cli_source_name(args.stock, args.sample, args.source) is not None:
            raise SystemExit("Do not pass SOURCE/--stock/--sample together with --all.")
        run_evaluation_all(
            no_plot=args.no_plot,
            predict=predict_fn,
            h=ph,
            algo_label=algo_label,
        )
        return

    name = cli_source_name(args.stock, args.sample, args.source)

    if name is not None:
        ensure_sample_exists(name)
        run_evaluation(name, predict=predict_fn, h=ph, algo_label=algo_label)
    else:
        default = "s1"
        ensure_sample_exists(default)
        run_evaluation(default, predict=predict_fn, h=ph, algo_label=algo_label)
