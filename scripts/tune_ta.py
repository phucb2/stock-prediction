"""Grid-search hyperparameters for SES.TechnicalAnalysisForecast.

Reports mean (rel, abs) across every sample_data/*.npy and prints the top
configurations that satisfy ``rel > 0`` and ``abs < 0.05``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import compute_eval_from_pv, load_data_path  # noqa: E402
from SES import TechnicalAnalysisForecast  # noqa: E402
from utils import list_sample_npy_files  # noqa: E402


def evaluate(model: TechnicalAnalysisForecast, paths: list[Path]) -> tuple[float, float, float, float]:
    rels: list[float] = []
    abses: list[float] = []
    for path in paths:
        P, V = load_data_path(path)
        _, _, den, rel = compute_eval_from_pv(P, V, predict=model.predict, h=5)
        rels.append(rel)
        abses.append(den)
    return (
        float(np.mean(rels)),
        float(np.mean(abses)),
        float(np.min(rels)),
        float(np.max(abses)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15, help="Print top-N configurations.")
    ap.add_argument("--coarse", action="store_true", help="Use a coarser grid for a fast pass.")
    args = ap.parse_args()

    paths = list_sample_npy_files()
    if not paths:
        raise SystemExit("No samples found")

    if args.coarse:
        ema_fast_grid = [5, 12]
        ema_slow_grid = [26, 50]
        bb_grid = [20]
        rsi_grid = [14]
        vol_grid = [20, 60]
        obv_grid = [20]
        cap_grid = [0.02, 0.05]
        w_trend_grid = [0.0, 0.1, 0.3]
        w_rsi_grid = [0.0, 0.1]
        w_bb_grid = [0.0, 0.1]
        w_obv_grid = [0.0, 0.1]
    else:
        # Coarse pass showed only the RSI mean-reversion term has predictive value.
        # Zoom in on rsi_period, w_rsi, vol_period and a small w_bb perturbation.
        ema_fast_grid = [12]
        ema_slow_grid = [26]
        bb_grid = [10, 20, 40]
        rsi_grid = [7, 10, 14, 21, 28]
        vol_grid = [10, 20, 60]
        obv_grid = [20]
        cap_grid = [0.05]
        w_trend_grid = [0.0, 0.02]
        w_rsi_grid = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
        w_bb_grid = [0.0, 0.02, 0.05]
        w_obv_grid = [0.0]

    results: list[tuple[float, float, float, float, dict]] = []
    grid = list(
        itertools.product(
            ema_fast_grid,
            ema_slow_grid,
            bb_grid,
            rsi_grid,
            vol_grid,
            obv_grid,
            cap_grid,
            w_trend_grid,
            w_rsi_grid,
            w_bb_grid,
            w_obv_grid,
        )
    )
    print(f"Trying {len(grid)} configurations across {len(paths)} samples...")

    for ef, es, bb, rsi, vp, op, cap, wt, wr, wb, wo in grid:
        if ef >= es:
            continue
        if wt == wr == wb == wo == 0.0:
            continue
        try:
            model = TechnicalAnalysisForecast(
                ema_fast=ef,
                ema_slow=es,
                rsi_period=rsi,
                bb_period=bb,
                obv_period=op,
                vol_period=vp,
                w_trend=wt,
                w_rsi=wr,
                w_bb=wb,
                w_obv=wo,
                cap=cap,
                use_volume=(wo > 0.0),
            )
        except ValueError:
            continue
        mean_rel, mean_abs, min_rel, max_abs = evaluate(model, paths)
        params = {
            "ema_fast": ef,
            "ema_slow": es,
            "bb_period": bb,
            "rsi_period": rsi,
            "vol_period": vp,
            "obv_period": op,
            "cap": cap,
            "w_trend": wt,
            "w_rsi": wr,
            "w_bb": wb,
            "w_obv": wo,
        }
        results.append((mean_rel, mean_abs, min_rel, max_abs, params))

    results.sort(key=lambda r: -r[0])
    print(f"\nTop {args.top} by mean rel:")
    for mean_rel, mean_abs, min_rel, max_abs, params in results[: args.top]:
        ok = "OK" if (mean_rel > 0 and mean_abs < 0.05) else "  "
        print(
            f"{ok}  mean_rel={mean_rel:+.4f}  mean_abs={mean_abs:.4f}  "
            f"min_rel={min_rel:+.4f}  max_abs={max_abs:.4f}  | {params}"
        )

    qualifying = [r for r in results if r[0] > 0 and r[1] < 0.05]
    print(f"\n{len(qualifying)}/{len(results)} configurations satisfy mean_rel>0 and mean_abs<0.05.")


if __name__ == "__main__":
    main()
