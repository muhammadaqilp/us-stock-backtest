"""Daily stock-pick report (manual research aid). Pure function over already-fetched, validated frames.

Pipeline: trim bars from sessions that are not yet complete -> reject tickers whose latest bar is not the benchmark's
latest completed bar -> indicators -> qualification (3 independent strategies) -> score/rank -> one row per ticker
(all matching setups kept) -> top N -> plan fields. Nothing is padded to reach N; zero picks is a valid outcome.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .data import expected_last_session
from .indicators import add_indicators
from .screener import build_scan_output, consolidate_by_ticker, rank_candidates

DISCLAIMER = ("Research aid only, not investment advice. Scores rank setups; they are not probabilities of profit. "
              "Historical walk-forward results (see README) show no demonstrated edge. Price levels are reference "
              "levels from the signal date, not guaranteed execution prices.")


class StaleMarketData(RuntimeError):
    """Benchmark data is older than the latest completed session; no picks are produced."""


def classify_report(rep) -> str:
    """Bucket a failed DataReport into a short exclusion reason."""
    text = "; ".join(rep.errors).lower()
    if "retrieval failed" in text or "empty" in text:
        return "data_retrieval_failed"
    if "insufficient history" in text and "non-positive" not in text and "inconsistent" not in text:
        return "insufficient_history"
    if "incomplete:" in text:
        return "incomplete_vs_benchmark_calendar"
    return "invalid_ohlcv"


def completed_market_bar(spy: pd.DataFrame, s, expected_bar_date=None):
    """(spy trimmed to completed sessions, latest completed bar date, expected date). Raises StaleMarketData if too old."""
    expected = expected_last_session() if expected_bar_date is None else pd.Timestamp(expected_bar_date)
    spy = spy.loc[spy.index <= expected]                      # drop a partial / future session bar
    if spy.empty:
        raise StaleMarketData("no completed SPY bar available")
    market_bar = spy.index.max()
    lag = int(np.busday_count(market_bar.date(), expected.date()))
    if lag > s.max_stale_sessions:
        raise StaleMarketData(f"SPY latest completed bar {market_bar.date()} is {lag} sessions behind expected "
                              f"{expected.date()}; refresh data")
    return spy, market_bar, expected


def build_daily_picks(frames: dict, spy: pd.DataFrame, reports: list, s, *, names: dict | None = None,
                      earnings_lookup=None, expected_bar_date: pd.Timestamp | None = None,
                      top_n: int | None = None, scan_time: datetime | None = None, verified_only: bool = False, news_lookup=None):
    """Returns (picks DataFrame, meta dict). Raises StaleMarketData if SPY's last completed bar is too old."""
    top_n = s.top_n if top_n is None else top_n
    spy, market_bar, expected = completed_market_bar(spy, s, expected_bar_date)
    excluded = Counter()
    for r in reports:
        if not r.ok:
            excluded[classify_report(r)] += 1
    current, rmap = {}, {r.ticker: r for r in reports}
    for t, f in frames.items():
        f = f.loc[f.index <= market_bar]            # common as-of date = SPY's newest valid bar (RS needs SPY)
        if f.empty or f.index.max() != market_bar:
            excluded["stale_or_missing_latest_bar"] += 1
        elif not (f["Volume"].iloc[-1] > 0):
            excluded["latest_bar_zero_volume"] += 1
        else:
            current[t] = add_indicators(f, s, spy)
    diag: dict = {}
    cands = rank_candidates(current, s, explain=True, diagnostics=diag)
    ranked = consolidate_by_ticker(cands)
    latest = {t: market_bar.date().isoformat() for t in current}
    # verified_only: check EVERY qualified ticker first, keep only those passing all mandatory checks, then cap at top_n.
    picks = build_scan_output(ranked if verified_only else ranked.head(top_n), rmap, latest, s,
                              earnings_lookup=earnings_lookup, names=names, news_lookup=news_lookup)
    unverified_dropped = {}
    if verified_only and len(picks):
        for reasons in picks.loc[picks["plan_status"] != "VERIFIED_PLAN", "unverified_because"]:
            for r in reasons.split(","):
                r = r.split("(")[0]
                unverified_dropped[r] = unverified_dropped.get(r, 0) + 1
        picks = picks[picks["plan_status"] == "VERIFIED_PLAN"].head(top_n).copy()
        picks["rank"] = range(1, len(picks) + 1)
        picks = picks.reset_index(drop=True)
    not_selected = {k: v for k, v in diag.items() if k != "qualified"}
    lag = int(np.busday_count(market_bar.date(), expected.date()))
    note = ("" if lag == 0 else f"Newest valid bar is {market_bar.date()} but the latest completed session should be "
            f"{expected.date()} ({lag} session(s) behind): picks reflect the {market_bar.date()} close. Re-run later after "
            "Yahoo finalizes the bar, or if that session was a US holiday this is expected.")
    meta = {
        "scan_time_utc": (scan_time or datetime.now(UTC)).isoformat(timespec="seconds"),
        "market_bar_date": market_bar.date().isoformat(),
        "expected_last_completed_session": expected.date().isoformat(),
        "bar_lag_sessions": lag, "data_note": note,
        "tickers_requested": len(reports),
        "tickers_examined": len(current),
        "tickers_excluded_data": int(sum(excluded.values())),
        "exclusion_reasons_data": dict(excluded),
        "examined_not_selected_reasons": not_selected,
        "tickers_qualified": int(diag.get("qualified", 0)),
        "picks_returned": len(picks),
        "verified_only": verified_only,
        "qualified_but_unverified_dropped": unverified_dropped,
        "max_picks": top_n, "min_score": s.min_score,
        "capital_usd": s.initial_capital,
        "price_label": "Reference levels from the signal-date close/ATR14; not guaranteed execution prices.",
        "disclaimer": DISCLAIMER,
    }
    return picks, meta
