"""Deterministic 0-100 ranking score. This is a RANKING heuristic, not a probability of profit.

Components (max points): trend 25, momentum 25, relative strength 20, volume 15, risk/volatility 15.
Each component is a piecewise-linear function clipped to [0,1] times its max points; ranges are hypotheses.
"""
import numpy as np
import pandas as pd

WEIGHTS = {"trend": 25, "momentum": 25, "relative_strength": 20, "volume": 15, "risk": 15}


def _lin(x, lo, hi):
    if x is None or not np.isfinite(x) or hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score_components(r, s) -> dict:
    close = float(r["Close"])
    trend = (8.0 * (close > r["sma50"]) + 8.0 * (r["sma50"] > r["sma200"])
             + 9.0 * _lin(r.get("sma50_slope20", np.nan), 0.0, 0.05))        # rising SMA50 (0..5% over 20 sessions)
    momentum = 25.0 * (0.6 * _lin(r["ret20"], 0.0, 0.15) + 0.4 * _lin(r["ret60"], 0.0, 0.30))
    rs_vals = [r.get(f"rs{n}", np.nan) for n in s.rs_lookbacks]
    rs_vals = [v for v in rs_vals if pd.notna(v)]
    rel = 20.0 * (np.mean([_lin(v, -0.05, 0.15) for v in rs_vals]) if rs_vals else 0.0)  # -5%..+15% vs SPY
    volume = 15.0 * _lin(r["rel_volume"], 1.0, 2.5)
    risk = 15.0 * (1.0 - _lin(r["atr_pct"], 0.02, 0.08))                      # calmer ATR% earns more
    comp = {"trend": trend, "momentum": momentum, "relative_strength": rel, "volume": volume, "risk": risk}
    return {k: round(float(v), 2) for k, v in comp.items()}


def score_row(r, s) -> tuple[float, dict]:
    comp = score_components(r, s)
    return round(sum(comp.values()), 2), comp
