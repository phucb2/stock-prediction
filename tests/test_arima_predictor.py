"""Tests for SES.StatsmodelsARIMAReturns (rolling ARIMA on one-step returns)."""

from __future__ import annotations

import unittest

import numpy as np

from SES import StatsmodelsARIMAReturns


class TestStatsmodelsARIMAReturns(unittest.TestCase):
    def test_shape_and_q0(self) -> None:
        P = np.array([100.0, 101.0, 102.0, 101.5], dtype=float)
        v = np.zeros(len(P))
        Q = StatsmodelsARIMAReturns(p=0, d=0, q=0, min_obs=1).predict(P, v)
        self.assertEqual(Q.shape, P.shape)
        self.assertEqual(Q[0], 0.0)

    def test_rejects_non_positive_prices(self) -> None:
        with self.assertRaises(ValueError):
            StatsmodelsARIMAReturns().predict(np.array([100.0, -1.0]), np.zeros(2), 5)

    def test_rejects_negative_order(self) -> None:
        with self.assertRaises(ValueError):
            StatsmodelsARIMAReturns(p=-1, d=0, q=0)
        with self.assertRaises(ValueError):
            StatsmodelsARIMAReturns(min_obs=0)

    def test_registry_name(self) -> None:
        from main import resolve_predictor_class

        self.assertIs(
            resolve_predictor_class("StatsmodelsARIMAReturns"),
            StatsmodelsARIMAReturns,
        )

    def test_last_point_matches_placeholder_rule(self) -> None:
        P = np.linspace(100.0, 110.0, 8)
        v = np.zeros(len(P))
        Q = StatsmodelsARIMAReturns(p=0, d=0, q=0, min_obs=1).predict(P, v)
        self.assertEqual(Q[-1], Q[-2])


if __name__ == "__main__":
    unittest.main()
