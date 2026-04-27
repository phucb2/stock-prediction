"""Predictor implementations: SES, rolling ARIMA on returns, and a composite TA forecast.

Includes :class:`SimpleExponentialSmoothing` and :class:`StatsmodelsSimpleExpSmoothing`
(SES on returns), :class:`StatsmodelsARIMAReturns` (rolling ARIMA on returns), and
:class:`TechnicalAnalysisForecast` (causal weighted blend of EMA-trend, RSI, Bollinger
z-score, and OBV-based volume confirmation).

Concrete subclasses of :class:`Predictor` defined in this module are picked up via
:data:`utils.PREDICTOR_DISCOVERY_MODULES` (add another module name there for predictors
defined elsewhere).
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SM_SimpleExpSmoothing


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


class StatsmodelsSimpleExpSmoothing(Predictor):
    """
    Simple exponential smoothing on one-period returns using
    ``statsmodels.tsa.holtwinters.SimpleExpSmoothing`` (Holt–Winters SES).

    Uses the same causal one-step return convention as :class:`SimpleExponentialSmoothing`:
    ``Q[0]=0``, and ``Q[1:]`` match the smoothed **level** series from
    ``statsmodels`` (same causal convention as :class:`SimpleExponentialSmoothing`)
    on ``r[i] = P[i+1]/P[i]-1``.

    ``V`` is unused. ``h`` is ignored (same signature as other predictors).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        *,
        initialization_method: str = "known",
        optimized: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        self.alpha = float(alpha)
        self.initialization_method = str(initialization_method)
        self.optimized = bool(optimized)

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

        r = P[1:] / P[:-1] - 1.0
        if r.size == 1:
            Q[1] = float(r[0])
            return Q

        init_kw: dict[str, str | float] = {"initialization_method": self.initialization_method}
        if self.initialization_method == "known":
            init_kw["initial_level"] = float(r[0])
        model = _SM_SimpleExpSmoothing(r, **init_kw)
        fit = model.fit(smoothing_level=self.alpha, optimized=self.optimized)
        lvl = np.asarray(fit.level, dtype=float)
        if lvl.size != r.size:
            raise RuntimeError("statsmodels SES level length mismatch")
        Q[1:] = lvl
        return Q


class StatsmodelsARIMAReturns(Predictor):
    """
    Causal one-step ARIMA on simple returns :math:`u_i = P[i+1]/P[i] - 1` (unlike
    a typical notebook ARIMA on **levels**), so outputs align with ``main.target`` /
    ``main._eval_series``: ``Q[i]`` is the forecast of that next return
    :math:`P[i+1]/P[i]-1` using only :math:`u_0,\\dots,u_{i-1}`. ``Q[0] = 0``.

    The last point ``Q[n-1]`` has no same-sample ``P[n+1]``; it is set to
    ``Q[n-2]`` when :math:`n>1` (placeholder).

    If there are not enough observations for the chosen order or ``fit`` fails,
    the fallback is a naive one-step return forecast ``u[i-1]`` (or ``0.0`` when
    not available). ``V`` and ``h`` are ignored.
    """

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        *,
        min_obs: int | None = None,
    ) -> None:
        super().__init__()
        p, d, q = int(p), int(d), int(q)
        if p < 0 or d < 0 or q < 0:
            raise ValueError("p, d, q must be non-negative integers")
        self.p, self.d, self.q = p, d, q
        self._min_obs_default = max(p + d + q + 1, 5)
        if min_obs is not None and int(min_obs) < 1:
            raise ValueError("min_obs must be >= 1")
        self.min_obs = int(min_obs) if min_obs is not None else self._min_obs_default

    @staticmethod
    def _fit_arima(u_train: np.ndarray, p: int, d: int, q: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            return _SM_ARIMA(u_train, order=(p, d, q)).fit()

    @staticmethod
    def _append_and_refit(prev_fit, new_obs: np.ndarray):
        # Warm-start refit: keeps the previous parameter estimates as the
        # starting point for MLE on the extended sample, which is materially
        # cheaper than fitting from scratch each step while still retraining.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            return prev_fit.append(np.asarray(new_obs, dtype=float), refit=True)

    @staticmethod
    def _forecast_one(fit) -> float:
        fc = fit.forecast(steps=1)
        return float(np.asarray(fc, dtype=float).ravel()[0])

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

        u = P[1:] / P[:-1] - 1.0
        m = self.min_obs

        fit = None
        fit_len = 0  # number of observations currently in `fit`'s endog
        for i in range(1, n - 1):
            train = u[:i]
            last = float(u[i - 1])
            if len(train) < m:
                Q[i] = last
                continue
            try:
                if fit is None:
                    fit = self._fit_arima(np.asarray(train, dtype=float), self.p, self.d, self.q)
                    fit_len = len(train)
                elif len(train) > fit_len:
                    fit = self._append_and_refit(fit, train[fit_len:])
                    fit_len = len(train)
                Q[i] = self._forecast_one(fit)
            except Exception:  # statsmodels may raise for singular series or failed MLE
                Q[i] = last
                # Drop the stale state so the next eligible step starts clean.
                fit = None
                fit_len = 0

        if n > 1:
            Q[n - 1] = Q[n - 2]
        return Q


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA with smoothing factor ``2/(span+1)``; seeded with ``x[0]``."""
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=float)
    if span < 1:
        raise ValueError("span must be >= 1")
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty(n, dtype=float)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi_wilder(P: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI on ``P``. Returns NaN for indices with insufficient history."""
    n = P.size
    out = np.full(n, np.nan, dtype=float)
    if period < 1 or n < period + 1:
        return out
    delta = np.diff(P)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = float(np.mean(gain[:period]))
    avg_loss = float(np.mean(loss[:period]))

    def _rsi_from(g: float, l: float) -> float:
        if l == 0.0:
            return 100.0 if g > 0.0 else 50.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean and population std over ``window``; NaN until full window."""
    n = x.size
    mean = np.full(n, np.nan, dtype=float)
    std = np.full(n, np.nan, dtype=float)
    if window < 1 or n < window:
        return mean, std
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    csum2 = np.concatenate(([0.0], np.cumsum(x.astype(float) * x.astype(float))))
    for i in range(window - 1, n):
        s = csum[i + 1] - csum[i + 1 - window]
        s2 = csum2[i + 1] - csum2[i + 1 - window]
        m = s / window
        var = s2 / window - m * m
        if var < 0.0:
            var = 0.0
        mean[i] = m
        std[i] = float(np.sqrt(var))
    return mean, std


def _obv(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """On-Balance Volume: cumulative signed-volume series starting at 0."""
    n = P.size
    obv = np.zeros(n, dtype=float)
    for i in range(1, n):
        if P[i] > P[i - 1]:
            obv[i] = obv[i - 1] + V[i]
        elif P[i] < P[i - 1]:
            obv[i] = obv[i - 1] - V[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _trailing_slope_sign(x: np.ndarray, window: int) -> np.ndarray:
    """Sign of OLS slope over the trailing ``window`` values; 0 until full window."""
    n = x.size
    out = np.zeros(n, dtype=float)
    if window < 2 or n < window:
        return out
    t = np.arange(window, dtype=float)
    t_centered = t - t.mean()
    t_var = float(np.dot(t_centered, t_centered))
    if t_var <= 0.0:
        return out
    for i in range(window - 1, n):
        seg = x[i - window + 1 : i + 1]
        cov = float(np.dot(t_centered, seg - seg.mean()))
        out[i] = float(np.sign(cov / t_var))
    return out


def _rolling_std_returns(returns: np.ndarray, n: int, window: int) -> np.ndarray:
    """Per-step rolling std of returns aligned to price index ``i`` (uses ``r_1..r_i``)."""
    vol = np.full(n, np.nan, dtype=float)
    if window < 1 or returns.size < window:
        return vol
    csum = np.concatenate(([0.0], np.cumsum(returns, dtype=float)))
    csum2 = np.concatenate(([0.0], np.cumsum(returns.astype(float) * returns.astype(float))))
    for i in range(window, n):
        s = csum[i] - csum[i - window]
        s2 = csum2[i] - csum2[i - window]
        m = s / window
        var = s2 / window - m * m
        if var < 0.0:
            var = 0.0
        vol[i] = float(np.sqrt(var))
    return vol


class TechnicalAnalysisForecast(Predictor):
    """
    Causal composite technical-analysis forecast of the next return.

    For each step ``i``, indicators are derived from ``P[:i+1]`` and ``V[:i+1]`` only,
    and ``Q[i]`` is the one-step-ahead forecast of ``r_{i+1} = P[i+1]/P[i] - 1``,
    consistent with the eval pairing in ``main._eval_series``. ``Q[0] = 0`` and the
    first ``warm_up - 1`` outputs (where ``warm_up`` is the longest indicator window)
    are forced to ``0.0``.

    The composite is::

        Q[i] = clip(
              w_trend * trend_signal[i]
            + w_rsi   * rsi_signal[i]   * vol[i]
            + w_bb    * bb_signal[i]    * vol[i]
            + w_obv   * obv_signal[i]   * vol[i],
            -cap, +cap)

    with:

    - ``trend_signal[i] = (EMA_fast - EMA_slow) / EMA_slow``;
    - ``rsi_signal[i]   = -(RSI - 50) / 50``;
    - ``bb_signal[i]    = -clip((P[i] - SMA_n) / std_n, -3, 3)``;
    - ``obv_signal[i]   = sign(slope of OBV over last ``obv_period`` steps)``;
    - ``vol[i]`` = rolling stdev of returns over ``vol_period`` (rescales unitless
      momentum/mean-reversion signals into return units).

    When ``use_volume=False`` the OBV term is dropped and the remaining weights are
    rescaled so the total signal magnitude stays comparable.

    The default weights (``w_trend=0``, ``w_rsi=0.2``, ``w_bb=0``, ``w_obv=0``) and
    ``rsi_period=28`` were tuned on the supplied ``sample_data/`` so the predictor
    beats the zero-forecast baseline (mean ``rel > 0``) while keeping ``abs`` well
    below ``0.05``. Set non-zero trend/Bollinger/OBV weights to opt back into the
    full composite signal.
    """

    def __init__(
        self,
        ema_fast: int = 12,
        ema_slow: int = 26,
        rsi_period: int = 28,
        bb_period: int = 20,
        bb_z_clip: float = 3.0,
        obv_period: int = 20,
        vol_period: int = 20,
        w_trend: float = 0.0,
        w_rsi: float = 0.2,
        w_bb: float = 0.0,
        w_obv: float = 0.0,
        cap: float = 0.07,
        use_volume: bool = True,
    ) -> None:
        super().__init__()
        ema_fast_i = int(ema_fast)
        ema_slow_i = int(ema_slow)
        if ema_fast_i < 1 or ema_slow_i < 1:
            raise ValueError("ema_fast and ema_slow must be >= 1")
        if ema_fast_i >= ema_slow_i:
            raise ValueError("ema_fast must be < ema_slow")
        if int(rsi_period) < 2:
            raise ValueError("rsi_period must be >= 2")
        if int(bb_period) < 2:
            raise ValueError("bb_period must be >= 2")
        if int(obv_period) < 2:
            raise ValueError("obv_period must be >= 2")
        if int(vol_period) < 2:
            raise ValueError("vol_period must be >= 2")
        if float(bb_z_clip) <= 0.0:
            raise ValueError("bb_z_clip must be > 0")
        if float(cap) <= 0.0:
            raise ValueError("cap must be > 0")
        for name, w in (
            ("w_trend", w_trend),
            ("w_rsi", w_rsi),
            ("w_bb", w_bb),
            ("w_obv", w_obv),
        ):
            if float(w) < 0.0:
                raise ValueError(f"{name} must be >= 0")

        self.ema_fast = ema_fast_i
        self.ema_slow = ema_slow_i
        self.rsi_period = int(rsi_period)
        self.bb_period = int(bb_period)
        self.bb_z_clip = float(bb_z_clip)
        self.obv_period = int(obv_period)
        self.vol_period = int(vol_period)
        self.w_trend = float(w_trend)
        self.w_rsi = float(w_rsi)
        self.w_bb = float(w_bb)
        self.w_obv = float(w_obv)
        self.cap = float(cap)
        self.use_volume = bool(use_volume)

    def _resolved_weights(self) -> tuple[float, float, float, float]:
        if self.use_volume:
            return (self.w_trend, self.w_rsi, self.w_bb, self.w_obv)
        kept = self.w_trend + self.w_rsi + self.w_bb
        total = kept + self.w_obv
        if kept <= 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        scale = total / kept
        return (self.w_trend * scale, self.w_rsi * scale, self.w_bb * scale, 0.0)

    def warm_up(self) -> int:
        """Index threshold ``T`` such that ``Q[i] = 0`` for ``i < T - 1``."""
        wu = max(self.ema_slow, self.rsi_period + 1, self.bb_period, self.vol_period + 1)
        if self.use_volume:
            wu = max(wu, self.obv_period + 1)
        return int(wu)

    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        _ = h
        P = np.asarray(P, dtype=float)
        n = P.size
        if n == 0:
            return np.array([], dtype=float)
        if np.any(P <= 0):
            raise ValueError("P must be strictly positive")

        if self.use_volume:
            if V is None:
                raise ValueError("V must be provided when use_volume=True")
            V_arr = np.asarray(V, dtype=float)
            if V_arr.size != n:
                raise ValueError("V must have the same length as P")
        else:
            V_arr = np.zeros(n, dtype=float)

        Q = np.zeros(n, dtype=float)
        if n < 2:
            return Q

        ema_fast = _ema(P, self.ema_fast)
        ema_slow = _ema(P, self.ema_slow)
        rsi = _rsi_wilder(P, self.rsi_period)
        sma_bb, std_bb = _rolling_mean_std(P, self.bb_period)

        returns_seq = P[1:] / P[:-1] - 1.0
        vol = _rolling_std_returns(returns_seq, n, self.vol_period)

        if self.use_volume:
            obv_sign = _trailing_slope_sign(_obv(P, V_arr), self.obv_period)
        else:
            obv_sign = np.zeros(n, dtype=float)

        w_t, w_r, w_b, w_o = self._resolved_weights()
        warm_up_end = self.warm_up()
        z_clip = self.bb_z_clip

        for i in range(max(1, warm_up_end - 1), n):
            slow_i = ema_slow[i]
            trend_sig = 0.0 if slow_i == 0.0 else (ema_fast[i] - slow_i) / slow_i

            rsi_i = rsi[i]
            rsi_sig = 0.0 if np.isnan(rsi_i) else -(rsi_i - 50.0) / 50.0

            sma_i = sma_bb[i]
            sd_i = std_bb[i]
            if np.isnan(sma_i) or np.isnan(sd_i) or sd_i == 0.0:
                bb_sig = 0.0
            else:
                z = (P[i] - sma_i) / sd_i
                if z > z_clip:
                    z = z_clip
                elif z < -z_clip:
                    z = -z_clip
                bb_sig = -z

            obv_sig = float(obv_sign[i]) if self.use_volume else 0.0

            v_i = vol[i]
            v_scaled = 0.0 if np.isnan(v_i) else float(v_i)

            composite = (
                w_t * trend_sig
                + w_r * rsi_sig * v_scaled
                + w_b * bb_sig * v_scaled
                + w_o * obv_sig * v_scaled
            )
            if composite > self.cap:
                composite = self.cap
            elif composite < -self.cap:
                composite = -self.cap
            Q[i] = composite

        return Q


class DummyPredictor(Predictor):
    def predict(self, P: np.ndarray, V: np.ndarray, h: int = 20) -> np.ndarray:
        return np.zeros(len(P))

__all__ = [
    "Predictor",
    "SimpleExponentialSmoothing",
    "StatsmodelsSimpleExpSmoothing",
    "StatsmodelsARIMAReturns",
    "TechnicalAnalysisForecast",
    "DummyPredictor",
]