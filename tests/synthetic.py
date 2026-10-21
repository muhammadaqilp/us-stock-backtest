"""SYNTHETIC test fixtures. Everything here is invented data for unit tests.
Nothing produced from these fixtures is a market result and must never be reported as one."""
from dataclasses import replace

import numpy as np
import pandas as pd

from swing_screener.config import Settings


def S(**kw) -> Settings:
    """Settings for tests: zero costs by default so arithmetic is exact; override per test."""
    base = dict(commission_bps_per_side=0.0, slippage_bps_per_side=0.0, initial_capital=100_000.0,
                max_holding_days=15, pause_seconds=0.0, min_history_rows=50)
    base.update(kw)
    return replace(Settings(), **base)


def frame(rows, start="2024-01-02", volume=1_000_000):
    """rows: list of (open, high, low, close). SYNTHETIC."""
    idx = pd.bdate_range(start, periods=len(rows))
    a = np.array(rows, dtype=float)
    return pd.DataFrame({"Open": a[:, 0], "High": a[:, 1], "Low": a[:, 2], "Close": a[:, 3],
                         "Volume": float(volume)}, index=idx)


def flat(n, price=100.0, start="2024-01-02"):
    return frame([(price, price, price, price)] * n, start=start)


def signal(date, tickers, atr=2.0, score=80.0, setup="breakout", close=None):
    rows = [{"ticker": t, "setup": setup, "score": score - i, "signal_date": date.date().isoformat(), "_atr": atr,
             **({} if close is None else {"close": close})} for i, t in enumerate(tickers)]
    return pd.DataFrame(rows)


def random_walk(n, seed, start="2019-01-02", drift=0.0008, vol=0.015, price=50.0):
    """SYNTHETIC geometric random walk with valid OHLCV."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = rng.normal(drift, vol, n)
    close = price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[price], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(2_000_000, 6_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
