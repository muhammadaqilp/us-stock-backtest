"""Causal indicators. Every value at row t uses only rows <= t (rolling windows, shift for prior-bar references).

Benchmark-relative values use SPY's own calendar; a stock date with no SPY bar yields NaN (never forward-filled),
so such a day cannot qualify for relative-strength logic.
"""
import numpy as np
import pandas as pd


def atr_wilder(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1, skipna=False)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(frame: pd.DataFrame, settings, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    d = frame.sort_index().copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    d["sma20"], d["sma50"], d["sma200"] = (c.rolling(n).mean() for n in (20, 50, 200))
    d["pb_ma"] = c.rolling(settings.pullback_ma_window).mean()
    d["sma50_slope20"] = d["sma50"] / d["sma50"].shift(20) - 1
    # Prior-N-session high: shift(1) excludes the current session from its own breakout threshold.
    d["prior_high"] = h.rolling(settings.breakout_lookback).max().shift(1)
    for n in sorted({20, 60, *settings.rs_lookbacks}):
        d[f"ret{n}"] = c.pct_change(n)
    d["avg_dollar_vol20"] = (c * v).rolling(20).mean()
    # Relative volume vs the PRIOR 20 sessions (current session excluded so a spike does not dilute its own baseline).
    d["rel_volume"] = v / v.rolling(20).mean().shift(1)
    d["atr14"] = atr_wilder(h, l, c, 14)
    d["atr_pct"] = d["atr14"] / c
    d["recent_high"] = h.rolling(settings.pullback_high_window).max()
    # Lowest (low / pullback-MA) over the last K sessions: <= 1+tol means price came to the average recently.
    d["pb_touch_ratio"] = (l / d["pb_ma"]).rolling(settings.pullback_lookback).min()
    # Depth of the pullback: lowest close in the last K sessions relative to the recent high.
    d["pb_depth"] = 1 - c.rolling(settings.pullback_lookback).min() / d["recent_high"]
    d["high_rs_window"] = h.rolling(settings.rs_high_window).max()
    d["prev_high"] = h.shift(1)
    d["prev_close"] = c.shift(1)
    for n in sorted({*settings.rs_lookbacks}):
        if benchmark is not None and len(benchmark):
            b = benchmark["Close"].pct_change(n).reindex(d.index)  # no ffill: missing SPY bar => NaN
            d[f"bench_ret{n}"] = b
            d[f"rs{n}"] = d[f"ret{n}"] - b
        else:
            d[f"bench_ret{n}"] = np.nan
            d[f"rs{n}"] = np.nan
    # Market regime: SPY close above its own N-day SMA (SPY calendar, no ffill). Only computed when the filter is on.
    if getattr(settings, "regime_sma", 0) > 0:
        if benchmark is not None and len(benchmark):
            bc = benchmark["Close"]
            above = (bc > bc.rolling(settings.regime_sma).mean()).where(bc.rolling(settings.regime_sma).count() == settings.regime_sma)
            d["bench_regime_ok"] = above.reindex(d.index)
        else:
            d["bench_regime_ok"] = np.nan
    return d
