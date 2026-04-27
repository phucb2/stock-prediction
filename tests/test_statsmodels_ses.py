"""Tests for SES.StatsmodelsSimpleExpSmoothing (statsmodels holtwinters SES)."""

from __future__ import annotations

import unittest

import numpy as np

from SES import SimpleExponentialSmoothing, StatsmodelsSimpleExpSmoothing


class TestStatsmodelsSimpleExpSmoothing(unittest.TestCase):
    def test_matches_hand_rolled_ses(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(15):
            n = int(rng.integers(2, 40))
            P = np.cumprod(np.concatenate([[100.0], 1 + rng.normal(0, 0.02, size=n - 1)]))
            alpha = float(rng.choice([0.2, 0.3, 0.5, 0.8, 1.0]))
            v = np.zeros(n)
            a = SimpleExponentialSmoothing(alpha=alpha).predict(P, v)
            b = StatsmodelsSimpleExpSmoothing(alpha=alpha).predict(P, v)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_alpha_invalid(self) -> None:
        with self.assertRaises(ValueError):
            StatsmodelsSimpleExpSmoothing(alpha=0.0)
        with self.assertRaises(ValueError):
            StatsmodelsSimpleExpSmoothing(alpha=1.5)

    def test_registry_name(self) -> None:
        from main import resolve_predictor_class

        self.assertIs(
            resolve_predictor_class("StatsmodelsSimpleExpSmoothing"),
            StatsmodelsSimpleExpSmoothing,
        )


if __name__ == "__main__":
    unittest.main()
