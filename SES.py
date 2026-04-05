"""Simple exponential smoothing (SES) on daily returns for price-only forecasts.

Concrete subclasses of :class:`Predictor` defined in this module are picked up via
:data:`utils.PREDICTOR_DISCOVERY_MODULES` (add another module name there for predictors
defined elsewhere).
"""

from __future__ import annotations

import numpy as np


class Predictor:
    """Base interface: same signature as ``main.prediction``."""

    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        raise NotImplementedError


class SimpleExponentialSmoothing(Predictor):
    """
    Causal SES on one-period returns r_i = P[i]/P[i-1] - 1.

    - ``Q[0]`` is 0 (no prior return), matching ``target``'s first element.
    - For ``i >= 1``, after observing r_i, the smoothed level s_i is updated and
      ``Q[i] = s_i`` is used as the one-step-ahead forecast of the *next* return
      (consistent with ``main._eval_series`` pairing).

    ``V`` is unused. ``h`` is ignored (kept for a common ``predict`` signature);
    use a subclass or wrapper if you need an ``h``-based warm-up.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        self.alpha = float(alpha)

    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        _ = V, h
        P = np.asarray(P, dtype=float)
        n = P.size
        if n == 0:
            return np.array([], dtype=float)
        if np.any(P <= 0):
            raise ValueError("P must be strictly positive")

        Q = np.zeros(n, dtype=float)
        if n == 1:
            return Q

        s = P[1] / P[0] - 1.0
        Q[1] = s
        for i in range(2, n):
            r_i = P[i] / P[i - 1] - 1.0
            s = self.alpha * r_i + (1.0 - self.alpha) * s
            Q[i] = s
        return Q
    
class DummyPredictor(Predictor):
    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        return np.zeros(len(P))

__all__ = ["Predictor", "SimpleExponentialSmoothing", "DummyPredictor"]