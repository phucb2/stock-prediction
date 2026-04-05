"""Unit tests for SES.SimpleExponentialSmoothing."""

from __future__ import annotations

import unittest

import numpy as np

from SES import SimpleExponentialSmoothing


class TestSimpleExponentialSmoothing(unittest.TestCase):
    def test_alpha_invalid(self) -> None:
        with self.assertRaises(ValueError):
            SimpleExponentialSmoothing(alpha=0.0)
        with self.assertRaises(ValueError):
            SimpleExponentialSmoothing(alpha=1.5)

    def test_empty_and_single_price(self) -> None:
        self.assertEqual(SimpleExponentialSmoothing().predict(np.array([]), np.array([])).size, 0)
        p = np.array([100.0])
        out = SimpleExponentialSmoothing(alpha=0.5).predict(p, np.array([]))
        np.testing.assert_array_equal(out, [0.0])

    def test_length_matches_P(self) -> None:
        p = np.array([100.0, 101.0, 102.0, 103.0])
        v = np.zeros_like(p)
        out = SimpleExponentialSmoothing(alpha=0.2).predict(p, v)
        self.assertEqual(len(out), len(p))

    def test_constant_price_all_zero_forecast_after_first_return_slot(self) -> None:
        p = np.ones(5) * 50.0
        out = SimpleExponentialSmoothing(alpha=0.4).predict(p, np.zeros(5))
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-15)

    def test_alpha_one_follows_last_return(self) -> None:
        # alpha=1 => s_i = r_i, so Q[i] = r_i for i>=1
        p = np.array([100.0, 110.0, 121.0])
        out = SimpleExponentialSmoothing(alpha=1.0).predict(p, np.zeros(3))
        r1 = 110.0 / 100.0 - 1.0
        r2 = 121.0 / 110.0 - 1.0
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], r1)
        np.testing.assert_allclose(out[2], r2)

    def test_alpha_small_smooths(self) -> None:
        p = np.array([100.0, 110.0, 110.0, 110.0], dtype=float)
        out = SimpleExponentialSmoothing(alpha=0.5).predict(p, np.zeros(4))
        r1 = 0.1
        r2 = 0.0
        s2 = 0.5 * r2 + 0.5 * r1
        r3 = 0.0
        s3 = 0.5 * r3 + 0.5 * s2
        np.testing.assert_allclose(out[1], r1)
        np.testing.assert_allclose(out[2], s2)
        np.testing.assert_allclose(out[3], s3)

    def test_non_positive_price_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimpleExponentialSmoothing().predict(np.array([1.0, -1.0]), np.array([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
