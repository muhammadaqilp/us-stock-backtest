"""On-demand trading plan for tickers the user types. Same data checks, rules, score, levels and sizing as `scan`.

A ticker that does not qualify still gets a rule-by-rule checklist showing exactly what failed; any levels shown for it
are HYPOTHETICAL (ATR arithmetic only) and explicitly not a signal. Research aid, not advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import earnings_status, event_risk_text, news_link
from .indicators import add_indicators
from .picks import DISCLAIMER, classify_report, completed_market_bar
from .risk import levels, size_position
from .scoring import score_row
from .screener import PRICE_LABEL, explain_setup
from .strategies import SETUPS, passes_common_filters


def _fin(*v):
    return all(pd.notna(x) and np.isfinite(x) for x in v)


def _rule(text, ok, value):
    return {"rule": text, "passed": bool(ok), "value": value}


def checklist(r, s) -> dict:
    """Rule-by-rule evaluation mirroring strategies.py (display only; a test asserts it agrees with the qualifiers).
    Missing/NaN inputs always FAIL a rule - never pass."""
    def gt(a, b):
        return _fin(a, b) and a > b

    def ge(a, b):
        return _fin(a, b) and a >= b

    def f(x, d=2):
        return "n/a" if not _fin(x) else round(float(x), d)

    c = r.get("Close", np.nan)
    common = [
        _rule("all indicators computable (enough history)", passes_common_filters(r, s)[1] != "incomplete_indicators", None),
        _rule(f"close >= ${s.min_price:g}", ge(c, s.min_price), f(c)),
        _rule(f"20d avg dollar volume >= ${s.min_avg_dollar_volume:,.0f}", ge(r.get("avg_dollar_vol20"), s.min_avg_dollar_volume),
              f(r.get("avg_dollar_vol20"), 0)),
        _rule(f"ATR% between 0 and {s.max_atr_pct:.0%}", _fin(r.get("atr_pct")) and 0 < r["atr_pct"] <= s.max_atr_pct,
              f(r.get("atr_pct"), 4)),
    ]
    trend = [_rule("close > SMA50", gt(c, r.get("sma50")), f"{f(c)} vs {f(r.get('sma50'))}"),
             _rule("SMA50 > SMA200", gt(r.get("sma50"), r.get("sma200")), f"{f(r.get('sma50'))} vs {f(r.get('sma200'))}")]
    pb_ma = r.get("pb_ma", np.nan)
    rs = {n: r.get(f"rs{n}", np.nan) for n in s.rs_lookbacks}
    near_high = _fin(c, r.get("high_rs_window")) and c >= r["high_rs_window"] * (1 - s.rs_max_distance_from_high)
    setups = {
        "breakout": [
            _rule(f"close > prior {s.breakout_lookback}-session high", gt(c, r.get("prior_high")),
                  f"{f(c)} vs {f(r.get('prior_high'))}"),
            _rule(f"relative volume >= {s.breakout_volume_multiple:g}x", ge(r.get("rel_volume"), s.breakout_volume_multiple),
                  f(r.get("rel_volume"))),
        ],
        "pullback": [
            _rule(f"low came within {s.pullback_tolerance:.0%} of SMA{s.pullback_ma_window} in last {s.pullback_lookback} sessions",
                  _fin(r.get("pb_touch_ratio")) and r["pb_touch_ratio"] <= 1 + s.pullback_tolerance, f(r.get("pb_touch_ratio"), 4)),
            _rule(f"pulled back >= {s.pullback_min_depth:.0%} from {s.pullback_high_window}-session high",
                  ge(r.get("pb_depth"), s.pullback_min_depth), f(r.get("pb_depth"), 4)),
            _rule("reversal bar: close > open", gt(c, r.get("Open")), f"{f(c)} vs {f(r.get('Open'))}"),
            _rule("reversal bar: close > prior day's high", gt(c, r.get("prev_high")), f"{f(c)} vs {f(r.get('prev_high'))}"),
            _rule(f"close back above SMA{s.pullback_ma_window}", gt(c, pb_ma), f"{f(c)} vs {f(pb_ma)}"),
        ],
        "relative_strength": [
            _rule("20d return > 0", gt(r.get("ret20"), 0), f(r.get("ret20"), 4)),
            *[_rule(f"beats SPY over {n}d by > {s.rs_min_outperformance:.0%}", gt(v, s.rs_min_outperformance), f(v, 4))
              for n, v in rs.items()],
            _rule(f"within {s.rs_max_distance_from_high:.0%} of {s.rs_high_window}-session high", near_high,
                  f"{f(c)} vs {f(r.get('high_rs_window'))}"),
            _rule("close > SMA20", gt(c, r.get("sma20")), f"{f(c)} vs {f(r.get('sma20'))}"),
            _rule(f"relative volume >= {s.general_relative_volume:g}x", ge(r.get("rel_volume"), s.general_relative_volume),
                  f(r.get("rel_volume"))),
        ],
    }
    return {"common": common, "uptrend": trend, **setups}


def setups_from_checklist(cl: dict) -> list:
    base = all(x["passed"] for x in cl["common"] + cl["uptrend"])
    return [k for k in SETUPS if base and all(x["passed"] for x in cl[k])]


def _unavailable(ticker, name, status, why, market_bar):
    return {"ticker": ticker, "company_name": name, "status": status, "plan_status": "NO_PLAN", "reasons": why,
            "market_bar_date": market_bar, "price_label": PRICE_LABEL}


def build_plan(ticker, frame, spy, report, s, *, name="", earnings_lookup=None, expected_bar_date=None,
               news_lookup=None) -> dict:
    spy, market_bar, expected = completed_market_bar(spy, s, expected_bar_date)
    mb = market_bar.date().isoformat()
    if report is None or not report.ok:
        why = "; ".join(report.errors) if report is not None else "no data report"
        return _unavailable(ticker, name, "DATA_UNAVAILABLE", f"{classify_report(report) if report else 'no_data'}: {why}", mb)
    f = frame.loc[frame.index <= expected]
    if f.empty or f.index.max() != market_bar:
        last = "none" if f.empty else f.index.max().date().isoformat()
        return _unavailable(ticker, name, "DATA_NOT_CURRENT", f"latest bar {last} is not the latest completed session {mb}", mb)
    if not (f["Volume"].iloc[-1] > 0):
        return _unavailable(ticker, name, "DATA_NOT_CURRENT", "latest bar has zero volume (incomplete)", mb)
    r = add_indicators(f, s, spy).iloc[-1]
    cl = checklist(r, s)
    hits = setups_from_checklist(cl)
    complete = passes_common_filters(r, s)[1] != "incomplete_indicators"
    score, comp = score_row(r, s) if complete else (None, {})
    lv = levels(float(r["Close"]), float(r["atr14"]), s) if complete else {"valid": False}
    qualified = bool(hits) and lv["valid"] and lv["rr_gross"] >= s.min_reward_risk
    if not hits:
        status = "NO_SETUP_TODAY"
    elif score is None or score < s.min_score or not qualified:
        status = "SETUP_BELOW_MIN_SCORE"
    else:
        status = "SETUP_QUALIFIED"
    out = {"ticker": ticker, "company_name": name, "status": status, "setups": hits,
           "signal_date": mb, "latest_bar_date": mb, "market_bar_date": mb, "checklist": cl,
           "score": score, **{f"score_{k}": v for k, v in comp.items()},
           "min_score_required": s.min_score, "price_label": PRICE_LABEL, "disclaimer": DISCLAIMER,
           "data_warnings": "; ".join(report.warnings)}
    for k in ("Close", "sma20", "sma50", "sma200", "ret20", "ret60", "rel_volume", "atr14", "atr_pct", "avg_dollar_vol20"):
        out[{"Close": "close", "avg_dollar_vol20": "avg_dollar_volume20"}.get(k, k)] = None if pd.isna(r[k]) else float(r[k])
    for n in s.rs_lookbacks:
        out[f"rs{n}_vs_spy"] = None if pd.isna(r.get(f"rs{n}")) else float(r[f"rs{n}"])
    earnings = earnings_lookup(ticker) if earnings_lookup else None
    sd = pd.Timestamp(mb)
    out.update(earnings_date="" if earnings is None else earnings.date().isoformat(),
               earnings_status=earnings_status(earnings, sd, s.max_holding_days),
               event_risk=event_risk_text(earnings, sd, s.max_holding_days))
    if complete and lv["valid"]:
        sz = size_position(s.initial_capital, s.initial_capital, float(r["Close"]), lv["stop"], s)
        out.update(ref_entry=round(float(r["Close"]), 4), stop=round(lv["stop"], 4), target=round(lv["target"], 4),
                   rr_gross=round(lv["rr_gross"], 2), rr_net_of_costs=lv["rr_net"], max_holding_sessions=s.max_holding_days,
                   shares=sz["shares"], position_value=round(sz["shares"] * float(r["Close"]), 2),
                   risk_at_stop_usd=round(sz["shares"] * sz.get("risk_per_share", 0), 2),
                   est_round_trip_cost_usd=round(sz["shares"] * float(r["Close"]) * s.cost_rate * 2, 2),
                   sizing_note="" if sz["shares"] else f"0 shares fit the limits (binding limit: {sz['reason']}); risk is never raised")
        out["levels_are"] = {"SETUP_QUALIFIED": "PLAN_REFERENCE_LEVELS",
                             "SETUP_BELOW_MIN_SCORE": "SETUP_MET_BUT_SCORE_BELOW_MINIMUM_NOT_A_SIGNAL"
                             }.get(status, "HYPOTHETICAL_NO_QUALIFYING_SETUP_NOT_A_SIGNAL")
        if hits:
            ex = {k: explain_setup(k, r, s, lv["stop"]) for k in hits}
            out["explanation"] = " | ".join(f"[{k}] {v[0]}" for k, v in ex.items())
            out["invalidation"] = " | ".join(f"[{k}] {v[1]}" for k, v in ex.items())
    out["news"] = (news_lookup(ticker) if news_lookup else
                   {"status": "NOT_CHECKED", "headlines": [], "link": news_link(ticker), "note": "news lookup disabled"})
    reasons = []
    if status != "SETUP_QUALIFIED":
        reasons.append(status.lower())
    if report.stale:
        reasons.append("stale_data")
    if out.get("shares", 0) < 1:
        reasons.append("zero_shares")
    if out["earnings_status"] == "IN_HOLDING_WINDOW":
        reasons.append("earnings_in_holding_window")
    elif out["earnings_status"] == "UNKNOWN":
        reasons.append("earnings_unverified")
    out["unverified_because"] = ",".join(reasons)
    out["plan_status"] = "VERIFIED_PLAN" if not reasons else ("RESEARCH_CANDIDATE" if status == "SETUP_QUALIFIED" else "NO_TRADE_SETUP")
    return out


def plan_steps(p: dict, s) -> list:
    """Plain-language plan lines for a ticker that has levels."""
    if "stop" not in p:
        return []
    lead = {"PLAN_REFERENCE_LEVELS": "",
            "SETUP_MET_BUT_SCORE_BELOW_MINIMUM_NOT_A_SIGNAL": "NOT A SIGNAL (setup rules met but score is below the minimum) - ",
            }.get(p["levels_are"], "HYPOTHETICAL (no qualifying setup; not a signal) - ")
    return [
        f"{lead}Reference entry: signal-date close {p['ref_entry']}. The real entry would be the NEXT session's open (unknown now); "
        f"re-base stop/target on that open using {s.atr_stop_multiple:g} and {s.atr_target_multiple:g} x ATR14 ({p['atr14']:.2f}).",
        f"{lead}Stop / invalidation: {p['stop']}. Target: {p['target']}. Gross reward/risk {p['rr_gross']} (net of costs {p['rr_net_of_costs']}).",
        f"Time exit: close the position at the end of session {p['max_holding_sessions']} if neither stop nor target was hit.",
        (f"Size for ${s.initial_capital:,.0f}: {p['shares']} share(s) = ${p['position_value']} at the reference price, risking about "
         f"${p['risk_at_stop_usd']} at the stop, est. round-trip cost ${p['est_round_trip_cost_usd']}." if p["shares"]
         else f"Size for ${s.initial_capital:,.0f}: {p['sizing_note']}."),
        f"Earnings: {p['event_risk']}",
    ]
