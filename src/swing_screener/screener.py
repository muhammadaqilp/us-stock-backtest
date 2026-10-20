"""Candidate generation = qualification (strategies) THEN ranking (scoring). Used by `scan` and by the backtester."""
import pandas as pd

from .events import earnings_status, event_risk_text, news_link
from .risk import levels, size_position
from .scoring import score_row
from .strategies import QUALIFIERS, SETUPS, passes_common_filters

COLUMNS = ["ticker", "setup", "signal_date", "latest_bar_date", "score", "score_components", "close"]


def explain_setup(setup: str, r, s, stop: float) -> tuple[str, str]:
    """(why it qualifies, what would invalidate it) for one setup, built only from the signal-day indicator row.
    Invalidation text is manual guidance; only the stop / target / time exits are modelled in the backtest."""
    c = float(r["Close"])
    trend = f"Uptrend (close {c:.2f} > SMA50 {r['sma50']:.2f} > SMA200 {r['sma200']:.2f})"
    if setup == "breakout":
        why = (f"{trend}; closed above the prior {s.breakout_lookback}-session high {r['prior_high']:.2f} "
               f"on {r['rel_volume']:.1f}x relative volume.")
        inv = f"Close back below the breakout level {r['prior_high']:.2f} (failed breakout) or stop {stop:.2f}."
    elif setup == "pullback":
        why = (f"{trend}; pulled back {r['pb_depth']:.1%} from the {s.pullback_high_window}-session high to within "
               f"{s.pullback_tolerance:.0%} of SMA{s.pullback_ma_window}, then showed a reversal bar "
               f"(close above open, above prior high and above the average).")
        inv = f"Close below SMA50 {r['sma50']:.2f}, or stop {stop:.2f}."
    else:
        rs = ", ".join(f"{r.get(f'rs{n}', float('nan')):+.1%} vs SPY over {n}d" for n in s.rs_lookbacks)
        dist = 1 - c / r["high_rs_window"]
        why = (f"{trend}; outperforming SPY ({rs}); {dist:.1%} below its {s.rs_high_window}-session high, above SMA20, "
               f"relative volume {r['rel_volume']:.1f}x.")
        inv = f"Relative strength vs SPY turns negative, close below SMA20 {r['sma20']:.2f}, or stop {stop:.2f}."
    return why, inv


def rank_candidates(indicators: dict, s, as_of: pd.Timestamp | None = None, dedupe: bool = False,
                    apply_min_score: bool = True, explain: bool = False, diagnostics: dict | None = None) -> pd.DataFrame:
    """Row per (ticker, qualifying setup) using ONLY the indicator row dated `as_of` (or each ticker's last row when None).
    Deterministic: sorted by score desc, then ticker, then setup order."""
    rows = []
    diag = diagnostics if diagnostics is not None else {}

    def count(key):
        diag[key] = diag.get(key, 0) + 1

    for ticker in sorted(indicators):
        d = indicators[ticker]
        if d.empty:
            count("no_data")
            continue
        if as_of is not None:
            if as_of not in d.index:
                count("no_bar_on_signal_date")
                continue
            r = d.loc[as_of]
        else:
            r = d.iloc[-1]
        ok, why_not = passes_common_filters(r, s)
        if not ok:
            count(f"filter:{why_not}")
            continue
        hits = [n for n in SETUPS if QUALIFIERS[n](r, s)[0]]
        if not hits:
            count("no_setup_qualified")
            continue
        score, comp = score_row(r, s)
        if apply_min_score and score < s.min_score:
            count("below_min_score")
            continue
        lv = levels(float(r["Close"]), float(r["atr14"]), s)
        if not lv["valid"] or lv["rr_gross"] < s.min_reward_risk:
            count("invalid_levels_or_reward_risk")
            continue
        count("qualified")
        for setup in hits:
            text = dict(zip(("explanation", "invalidation"), explain_setup(setup, r, s, lv["stop"]))) if explain else {}
            rows.append({**text, "ticker": ticker, "setup": setup, "signal_date": r.name.date().isoformat(),
                         "score": score, "score_components": comp, "close": float(r["Close"]),
                         "sma20": r["sma20"], "sma50": r["sma50"], "sma200": r["sma200"],
                         "ret20": r["ret20"], "ret60": r["ret60"],
                         **{f"rs{n}": r.get(f"rs{n}") for n in s.rs_lookbacks},
                         "avg_dollar_volume20": r["avg_dollar_vol20"], "rel_volume": r["rel_volume"],
                         "atr14": r["atr14"], "atr_pct": r["atr_pct"], "_atr": float(r["atr14"])})
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    df["_o"] = df["setup"].map({n: i for i, n in enumerate(SETUPS)})
    df = df.sort_values(["score", "ticker", "_o"], ascending=[False, True, True]).drop(columns="_o")
    if dedupe:
        df = df.drop_duplicates("ticker", keep="first")
    return df.reset_index(drop=True)


def consolidate_by_ticker(cands: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker for the main ranked list, preserving every matching setup. The score is a property of the
    ticker's signal-day bar, so all setups share it; the primary setup is the first in SETUPS order."""
    if cands.empty:
        return cands.assign(setups=[], explanations=[], invalidations=[])
    rows = []
    for _, g in cands.groupby("ticker", sort=False):
        base = g.iloc[0].to_dict()
        base["setups"] = list(g["setup"])
        base["explanations"] = dict(zip(g["setup"], g["explanation"])) if "explanation" in g else {}
        base["invalidations"] = dict(zip(g["setup"], g["invalidation"])) if "invalidation" in g else {}
        rows.append(base)
    df = pd.DataFrame(rows)
    return df.sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _news_fields(ticker, news_lookup) -> dict:
    n = news_lookup(ticker) if news_lookup else None
    if n is None:
        return {"news_status": "NOT_CHECKED", "news_headlines": "", "news_link": news_link(ticker)}
    return {"news_status": n["status"], "news_headlines": " | ".join(
        f"{h['title']} ({h['publisher']}, {h['published'][:10]})" for h in n["headlines"]), "news_link": n["link"]}


PRICE_LABEL = ("Reference levels derived from the signal-date close and ATR14; not guaranteed or recommended execution prices. "
               "The real entry is unknown until the next session opens.")


def build_scan_output(cands: pd.DataFrame, data_reports: dict, latest_bar: dict, s, earnings_lookup=None,
                      names: dict | None = None, news_lookup=None) -> pd.DataFrame:
    """Attach plan fields to (consolidated) candidates. plan_status is VERIFIED_PLAN only if EVERY mandatory check passes:
    data validated and not stale, earnings date known and outside the holding window, >= 1 share fits the limits, net
    reward/risk > 0. Anything else is RESEARCH_CANDIDATE with the failed checks listed."""
    out = []
    for rank, (_, c) in enumerate(cands.iterrows(), start=1):
        setups = list(c["setups"]) if "setups" in c and isinstance(c["setups"], (list, tuple)) else [c["setup"]]
        expl = c["explanations"] if "explanations" in c and isinstance(c["explanations"], dict) else {}
        inv = c["invalidations"] if "invalidations" in c and isinstance(c["invalidations"], dict) else {}
        rep = data_reports.get(c["ticker"])
        lv = levels(c["close"], c["_atr"], s)
        sz = size_position(s.initial_capital, s.initial_capital, c["close"], lv["stop"], s)
        rt_cost = sz["shares"] * c["close"] * s.cost_rate * 2
        earnings = earnings_lookup(c["ticker"]) if earnings_lookup else None
        event = event_risk_text(earnings, pd.Timestamp(c["signal_date"]), s.max_holding_days)
        e_status = earnings_status(earnings, pd.Timestamp(c["signal_date"]), s.max_holding_days)
        reasons = []
        if rep is None or not rep.ok:
            reasons.append("data_validation_failed")
        if rep is not None and rep.stale:
            reasons.append("stale_data")
        if sz["shares"] < 1:
            reasons.append(f"zero_shares({sz['reason']})")
        if lv["rr_net"] <= 0:
            reasons.append("non_positive_net_rr")
        if e_status == "IN_HOLDING_WINDOW":
            reasons.append("earnings_in_holding_window")
        elif e_status == "UNKNOWN":
            reasons.append("earnings_unverified")
        warnings = list(rep.warnings) if rep is not None else []
        out.append({
            "rank": rank, "ticker": c["ticker"], "company_name": (names or {}).get(c["ticker"], ""),
            "setup": setups[0], "setups": setups,
            "plan_status": "RESEARCH_CANDIDATE" if reasons else "VERIFIED_PLAN",
            "unverified_because": ",".join(reasons),
            "explanation": " | ".join(f"[{k}] {expl[k]}" for k in setups if k in expl),
            "invalidation": " | ".join(f"[{k}] {inv[k]}" for k in setups if k in inv),
            "signal_date": c["signal_date"], "latest_bar_date": latest_bar[c["ticker"]],
            "score": c["score"], **{f"score_{k}": v for k, v in c["score_components"].items()},
            "close": round(c["close"], 4),
            "sma20": round(c["sma20"], 4), "sma50": round(c["sma50"], 4), "sma200": round(c["sma200"], 4),
            "ret20_pct": round(c["ret20"] * 100, 2), "ret60_pct": round(c["ret60"] * 100, 2),
            **{f"rs{n}_vs_spy_pct": round(c[f"rs{n}"] * 100, 2) for n in s.rs_lookbacks},
            "avg_dollar_volume20": round(c["avg_dollar_volume20"]), "rel_volume": round(c["rel_volume"], 2),
            "atr14": round(c["atr14"], 4), "atr_pct": round(c["atr_pct"] * 100, 2),
            "entry_rule": "Reference entry = signal-date close; backtest convention is the next session's open, with stop "
                          "and target re-based on the actual open using the same ATR multiples.",
            "ref_entry": round(c["close"], 4), "stop": round(lv["stop"], 4), "target": round(lv["target"], 4),
            "rr_gross": round(lv["rr_gross"], 2), "rr_net_of_costs": lv["rr_net"],
            "max_holding_sessions": s.max_holding_days,
            "shares": sz["shares"], "position_value": round(sz["shares"] * c["close"], 2),
            "risk_at_stop_usd": round(sz["shares"] * sz.get("risk_per_share", 0), 2),
            "est_round_trip_cost_usd": round(rt_cost, 2),
            "earnings_date": "" if earnings is None else earnings.date().isoformat(),
            "earnings_status": e_status, "event_risk": event,
            "data_warnings": "; ".join(warnings),
            "price_label": PRICE_LABEL,
            **_news_fields(c["ticker"], news_lookup),
        })
    return pd.DataFrame(out)
