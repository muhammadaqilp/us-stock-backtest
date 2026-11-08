"""Strategy QUALIFICATION only (does the setup exist on this bar?). Ranking/scoring lives in scoring.py.

Each qualifier takes one indicator row (a pandas Series as of the signal-day close) and returns
(qualified: bool, detail: dict). Rule rationale is in each docstring and in the README.
"""
import numpy as np
import pandas as pd

SETUPS = ("breakout", "pullback", "relative_strength")


def _finite(*vals):
    return all(pd.notna(v) and np.isfinite(v) for v in vals)


def passes_common_filters(r, s) -> tuple[bool, str]:
    """Data-completeness, price, liquidity, volatility filters shared by every setup.
    Rationale: avoid illiquid/penny names where $1k-account costs and gaps dominate, and require every
    indicator to be computable (no silent NaN treated as pass)."""
    need = ("Close", "sma50", "sma200", "avg_dollar_vol20", "rel_volume", "atr14", "atr_pct", "ret20", "ret60")
    if not _finite(*(r.get(k, np.nan) for k in need)):
        return False, "incomplete_indicators"
    if r["Close"] < s.min_price:
        return False, "price_below_min"
    if r["avg_dollar_vol20"] < s.min_avg_dollar_volume:
        return False, "illiquid"
    if not (0 < r["atr_pct"] <= s.max_atr_pct):
        return False, "volatility_filter"
    if s.regime_sma > 0 and not regime_ok(r):
        return False, "market_regime"
    return True, ""


def regime_ok(r) -> bool:
    """SPY closed above its regime SMA on this date. Missing/unknown counts as NOT ok (never a silent pass)."""
    v = r.get("bench_regime_ok", np.nan)
    return bool(pd.notna(v) and v)


def established_uptrend(r) -> bool:
    """Close > SMA50 > SMA200: price above the intermediate trend and the intermediate above the long trend."""
    return bool(_finite(r["Close"], r["sma50"], r["sma200"]) and r["Close"] > r["sma50"] > r["sma200"])


def qualify_breakout(r, s):
    """Uptrend + close above the PRIOR N-session high (current session excluded) + relative volume >= threshold.
    Rationale: a close through a recent range on above-normal participation is the classic momentum-continuation trigger."""
    if not established_uptrend(r):
        return False, {"reason": "no_uptrend"}
    if not _finite(r["prior_high"], r["rel_volume"]):
        return False, {"reason": "incomplete"}
    ok = r["Close"] > r["prior_high"] and r["rel_volume"] >= s.breakout_volume_multiple
    return bool(ok), {"prior_high": float(r["prior_high"]), "rel_volume": float(r["rel_volume"])}


def qualify_pullback(r, s):
    """Uptrend + a real pullback (within the last K sessions the low reached within `tolerance` of the pullback MA AND
    the close was >= `min_depth` below the recent high) + REVERSAL CONFIRMATION today:
    close > open (bullish bar), close > prior session's high, and close back above the pullback MA.
    Rationale: proximity to an average alone is not a signal; requiring a higher-high bullish close shows buyers
    reasserting after the dip."""
    if not established_uptrend(r):
        return False, {"reason": "no_uptrend"}
    if not _finite(r["pb_touch_ratio"], r["pb_depth"], r["pb_ma"], r["prev_high"], r["Open"]):
        return False, {"reason": "incomplete"}
    touched = r["pb_touch_ratio"] <= 1 + s.pullback_tolerance
    deep = r["pb_depth"] >= s.pullback_min_depth
    reversal = r["Close"] > r["Open"] and r["Close"] > r["prev_high"] and r["Close"] > r["pb_ma"]
    return bool(touched and deep and reversal), {
        "touched_ma": bool(touched), "pullback_depth": float(r["pb_depth"]), "reversal_confirmed": bool(reversal)}


def qualify_relative_strength(r, s):
    """Uptrend + positive 20d momentum + outperformance vs SPY over EVERY configured lookback by more than the margin
    + continuation: close within `rs_max_distance_from_high` of the N-session high, above SMA20, on at least
    `general_relative_volume` relative volume.
    Rationale: leaders that stay near highs while the market lags tend to be where persistent demand is; the
    near-high/above-SMA20 condition filters out leaders that are already rolling over."""
    if not established_uptrend(r):
        return False, {"reason": "no_uptrend"}
    rs_vals = [r.get(f"rs{n}", np.nan) for n in s.rs_lookbacks]
    if not _finite(r["ret20"], r["sma20"], r["high_rs_window"], r["rel_volume"], *rs_vals):
        return False, {"reason": "incomplete"}
    outperf = all(v > s.rs_min_outperformance for v in rs_vals)
    near_high = r["Close"] >= r["high_rs_window"] * (1 - s.rs_max_distance_from_high)
    cont = near_high and r["Close"] > r["sma20"] and r["rel_volume"] >= s.general_relative_volume
    return bool(r["ret20"] > 0 and outperf and cont), {
        "outperforms_all_lookbacks": bool(outperf), "near_high": bool(near_high)}


QUALIFIERS = {"breakout": qualify_breakout, "pullback": qualify_pullback, "relative_strength": qualify_relative_strength}


def qualify_all(r, s) -> list[str]:
    """Setups that qualify for this row (empty if common filters fail)."""
    ok, _ = passes_common_filters(r, s)
    if not ok:
        return []
    return [name for name, fn in QUALIFIERS.items() if name in s.enabled_setups and fn(r, s)[0]]
