"""Stock prediction assignment — core data loading, prediction, and evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

# authors
# Nguyen Van A, ...

# warning:
# only customize the 3-4-5 blocks

from utils import (
    DefaultPredictor,
    make_predictor,
    parse_algo_params,
    registered_predictor_names,
    resolve_pv_path,
    resolve_predictor_class,
)


def _load_pv_array(stk: str) -> np.ndarray:
    return np.load(resolve_pv_path(stk), allow_pickle=True)


def _split_pv(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return A[:, 2], A[:, 6]


def load_data(stk: str) -> tuple[np.ndarray, np.ndarray]:
    """Input your data here (notebook cell 3)."""
    A = _load_pv_array(stk)
    return _split_pv(A)


def load_data_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    A = np.load(path, allow_pickle=True)
    return _split_pv(A)


def prediction_default(P, V, h=20):
    """Template predictor (notebook cell 4): truncated MA-style return forecast."""

    def truncate(u):
        v = u
        if v < -0.07:
            u = -0.07
        elif v > 0.07:
            u = 0.07
        return u

    n, L, Q = len(P), [], []
    for i in range(n):
        s, cnt = P[i], 1
        while (cnt <= h) and (i - cnt >= 0):
            s += P[i - cnt]
            cnt += 1
        L.append(s / cnt)
    for i in range(n):
        tmp = L[i] / P[i] - 1
        tmp = truncate(tmp)
        Q.append(tmp)
    return Q


def prediction(P, V, h=20):
    """Customize your prediction (notebook cell 4); delegates to the default template."""
    return prediction_default(P, V, h)


def target(P, V):
    """Keep the core function unchanged (notebook cell 5)."""
    n, Q = len(P), [0]
    for i in range(1, n):
        Q.append(P[i] / P[i - 1] - 1)
    return Q


def _eval_series(p, t):
    p, t = p[1:], t[1:]
    n, e, f = len(t), [], []
    for i in range(1, n):
        e.append(t[i] - p[i - 1])
        f.append(t[i])
    e = np.asarray(e, dtype=float)
    f = np.asarray(f, dtype=float)
    den = np.nanquantile(np.abs(e), 0.5) + 0.5 * np.nanquantile(np.abs(e), 0.9)
    num = np.nanquantile(np.abs(f), 0.5) + 0.5 * np.nanquantile(np.abs(f), 0.9)
    return e, f, den, num


def evaluate(p, t, dspl=False):
    _, _, den, num = _eval_series(p, t)
    if dspl:
        print(
            f"\n\tbase = {round(num, 3)}  |  abs = {round(den, 3)}  |  rel = {round(1 - den / num, 3)}\n"
        )
        return None
    return den, 1 - den / num


def compute_eval_from_pv(
    P: np.ndarray,
    V: np.ndarray,
    *,
    predict: Callable[..., np.ndarray] | None = None,
    h: int = 5,
) -> tuple[np.ndarray, float, float, float]:
    """Returns (errors e, base, abs metric, rel)."""
    predict_fn = predict if predict is not None else prediction_default
    p, t = predict_fn(P, V, h), target(P, V)
    e, _, den, num = _eval_series(p, t)
    rel = 1 - den / num if num != 0 else float("nan")
    return e, float(num), float(den), float(rel)


if __name__ == "__main__":
    from utils import main_cli

    main_cli()
