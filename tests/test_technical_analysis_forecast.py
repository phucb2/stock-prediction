"""Tests for SES.TechnicalAnalysisForecast (composite TA forecast on returns)."""

from __future__ import annotations

import unittest

import numpy as np

from SES import TechnicalAnalysisForecast


def _make_prices(rng: np.random.Generator, n: int, mu: float = 0.0, sigma: float = 0.01) -> np.ndarray:
    rets = rng.normal(mu, sigma, size=n - 1)
    return np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + rets)])


class TestTechnicalAnalysisForecast(unittest.TestCase):
    def test_shape_q0_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        n = 80
        P = _make_prices(rng, n)
        V = rng.uniform(1e3, 1e6, size=n)
        Q = TechnicalAnalysisForecast().predict(P, V)
        self.assertEqual(Q.shape, P.shape)
        self.assertEqual(Q.dtype, np.float64)
        self.assertEqual(Q[0], 0.0)

    def test_causality(self) -> None:
        rng = np.random.default_rng(1)
        n = 100
        P = _make_prices(rng, n)
        V = rng.uniform(1e3, 1e6, size=n)
        model = TechnicalAnalysisForecast()
        Q_full = model.predict(P, V)

        cut = 60
        P_alt = P.copy()
        V_alt = V.copy()
        # Replace strictly-future values; must not affect Q[:cut+1].
        P_alt[cut + 1 :] = P[cut] * (1.0 + rng.normal(0, 0.05, size=n - cut - 1))
        V_alt[cut + 1 :] = rng.uniform(1e3, 1e6, size=n - cut - 1)
        Q_alt = model.predict(P_alt, V_alt)
        np.testing.assert_allclose(Q_full[: cut + 1], Q_alt[: cut + 1], rtol=0, atol=1e-12)

    def test_bounds_respect_cap(self) -> None:
        rng = np.random.default_rng(2)
        n = 120
        P = _make_prices(rng, n, sigma=0.05)
        V = rng.uniform(1e3, 1e6, size=n)
        cap = 0.04
        Q = TechnicalAnalysisForecast(cap=cap).predict(P, V)
        self.assertTrue(np.all(np.abs(Q) <= cap + 1e-12))

    def test_warm_up_zeros(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        P = _make_prices(rng, n)
        V = rng.uniform(1e3, 1e6, size=n)
        model = TechnicalAnalysisForecast()
        Q = model.predict(P, V)
        wu = model.warm_up()
        np.testing.assert_array_equal(Q[: wu - 1], np.zeros(wu - 1, dtype=float))

    def test_trend_dominance_on_monotone_up(self) -> None:
        # Pure trend signal: monotonically rising prices should yield positive forecasts.
        n = 80
        P = np.linspace(100.0, 200.0, n)
        V = np.ones(n)
        model = TechnicalAnalysisForecast(
            w_trend=1.0, w_rsi=0.0, w_bb=0.0, w_obv=0.0, use_volume=False
        )
        Q = model.predict(P, V)
        post = Q[model.warm_up() :]
        self.assertGreater(float(post.mean()), 0.0)
        self.assertTrue(np.all(post >= 0.0))

    def test_mean_reversion_on_spike(self) -> None:
        # Pure Bollinger-mean-reversion forecast: an upward price spike well above
        # the recent mean should give a negative forecast at the spike index.
        n = 60
        P = np.full(n, 100.0)
        spike_idx = 45
        P[spike_idx:] = 130.0  # one big upward shift
        V = np.ones(n)
        model = TechnicalAnalysisForecast(
            w_trend=0.0,
            w_rsi=0.0,
            w_bb=1.0,
            w_obv=0.0,
            use_volume=False,
            bb_period=20,
            vol_period=20,
        )
        Q = model.predict(P, V)
        self.assertLess(Q[spike_idx], 0.0)

    def test_use_volume_false_ignores_V(self) -> None:
        rng = np.random.default_rng(4)
        n = 80
        P = _make_prices(rng, n)
        V_a = rng.uniform(1e3, 1e6, size=n)
        V_b = rng.uniform(1e3, 1e6, size=n)
        model = TechnicalAnalysisForecast(use_volume=False)
        Q_a = model.predict(P, V_a)
        Q_b = model.predict(P, V_b)
        np.testing.assert_allclose(Q_a, Q_b, rtol=0, atol=1e-12)

    def test_rejects_non_positive_prices(self) -> None:
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast().predict(np.array([100.0, -1.0]), np.ones(2))

    def test_rejects_volume_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast(use_volume=True).predict(
                np.array([100.0, 101.0, 102.0]), np.ones(2)
            )

    def test_rejects_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast(ema_fast=20, ema_slow=10)
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast(rsi_period=1)
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast(cap=0.0)
        with self.assertRaises(ValueError):
            TechnicalAnalysisForecast(w_trend=-0.1)

    def test_short_series_returns_zeros(self) -> None:
        Q = TechnicalAnalysisForecast().predict(np.array([100.0]), np.array([1.0]))
        np.testing.assert_array_equal(Q, np.zeros(1))

    def test_registry_name(self) -> None:
        from main import resolve_predictor_class

        self.assertIs(
            resolve_predictor_class("TechnicalAnalysisForecast"),
            TechnicalAnalysisForecast,
        )

    def test_default_params_meet_eval_targets(self) -> None:
        # Pins the tuning goal: with defaults, the predictor must beat the
        # zero-forecast baseline (mean rel > 0) and keep mean abs well below
        # 0.05 across every sample_data/*.npy. Skip when sample data is absent.
        import os

        from main import compute_eval_from_pv, load_data_path
        from utils import list_sample_npy_files

        if os.environ.get("STOCK_PV_NPY"):
            self.skipTest("STOCK_PV_NPY override active; skip global eval test")
        paths = list_sample_npy_files()
        if not paths:
            self.skipTest("No sample_data/*.npy available")

        model = TechnicalAnalysisForecast()
        rels: list[float] = []
        abses: list[float] = []
        for path in paths:
            P, V = load_data_path(path)
            _, _, den, rel = compute_eval_from_pv(P, V, predict=model.predict, h=5)
            rels.append(rel)
            abses.append(den)

        mean_rel = float(np.mean(rels))
        mean_abs = float(np.mean(abses))
        self.assertGreater(mean_rel, 0.0, f"mean rel = {mean_rel}")
        self.assertLess(mean_abs, 0.05, f"mean abs = {mean_abs}")
        self.assertLess(max(abses), 0.05, f"per-stock max abs = {max(abses)}")


if __name__ == "__main__":
    unittest.main()
