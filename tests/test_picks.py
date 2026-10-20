"""Daily-pick report tests. All frames are SYNTHETIC; nothing here is a market result."""
import json

import numpy as np
import pandas as pd
import pytest

from swing_screener.cli import main
from swing_screener.data import DataReport
from swing_screener.indicators import add_indicators
from swing_screener.picks import StaleMarketData, build_daily_picks, classify_report
from swing_screener.screener import (
    build_scan_output,
    consolidate_by_ticker,
    rank_candidates,
)
from tests.synthetic import S

N = 300
IDX = pd.bdate_range("2023-01-02", periods=N)
LAST = IDX[-1]
PS = dict(initial_capital=1000.0, min_score=0.0, min_avg_dollar_volume=1e6, min_price=5.0, top_n=10)


def uptrend(drift=0.003, jump=0.04, vol_mult=3.0, idx=IDX, price=50.0, breakout=True, volume=5e6):
    """Smooth synthetic uptrend; the last bar optionally gaps up on heavy volume (a breakout)."""
    c = price * np.cumprod(np.full(len(idx), 1 + drift))
    if breakout:
        c[-1] = c[-2] * (1 + jump)
    o = np.concatenate([[price], c[:-1]])
    v = np.full(len(idx), volume)
    if breakout:
        v[-1] = volume * vol_mult
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.002, "Low": np.minimum(o, c) * 0.998,
                         "Close": c, "Volume": v}, index=idx)


def spy_frame(idx=IDX):
    c = 400 * np.cumprod(np.full(len(idx), 1.0003))
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": 5e7}, index=idx)


def ok_reports(tickers):
    return [DataReport(t, status="OK", rows=N) for t in tickers]


def run(frames, s=None, **kw):
    s = s or S(**PS)
    kw.setdefault("expected_bar_date", LAST)
    kw.setdefault("earnings_lookup", lambda t: pd.Timestamp("2024-12-01"))
    return build_daily_picks(frames, spy_frame(), ok_reports(frames), s, **kw)


# ---------------------------------------------------------------- ranking / top-N / no padding
def test_ranking_is_descending_capped_deterministic_and_ranked():
    frames = {f"T{i:02d}": uptrend(drift=0.002 + 0.0002 * i, jump=0.02 + 0.004 * i) for i in range(12)}
    p1, m1 = run(frames)
    p2, _ = run(frames)
    assert m1["tickers_qualified"] == 12 and len(p1) == 10 == m1["picks_returned"]
    assert list(p1["rank"]) == list(range(1, 11))
    assert list(p1["score"]) == sorted(p1["score"], reverse=True)
    assert p1["ticker"].is_unique
    pd.testing.assert_frame_equal(p1, p2)                      # reproducible


def test_top_n_is_a_ceiling_not_a_target():
    frames = {"UP1": uptrend(), "UP2": uptrend(drift=0.0025),
              **{f"FLAT{i}": uptrend(drift=0.0, breakout=False) for i in range(3)}}
    p, m = run(frames)
    assert len(p) == 2 and set(p["ticker"]) == {"UP1", "UP2"}  # flat names are not added to fill 10
    assert sum(m["examined_not_selected_reasons"].values()) == 3 and m["tickers_examined"] == 5


def test_min_score_filters_instead_of_padding():
    p, m = run({"UP": uptrend()}, S(**{**PS, "min_score": 99.9}))
    assert p.empty and m["examined_not_selected_reasons"].get("below_min_score") == 1


# ---------------------------------------------------------------- duplicate tickers
def test_duplicate_setups_consolidated_into_one_row_preserving_all_setups():
    s = S(**PS)
    ind = {"UP": add_indicators(uptrend(), s, spy_frame())}
    raw = rank_candidates(ind, s, explain=True)
    assert len(raw) >= 2 and set(raw.ticker) == {"UP"}          # one row per setup before consolidation
    merged = consolidate_by_ticker(raw)
    assert len(merged) == 1 and merged.loc[0, "setups"] == list(raw["setup"]) and len(merged.loc[0, "setups"]) >= 2
    p, _ = run({"UP": uptrend()})
    assert len(p) == 1 and len(p.loc[0, "setups"]) >= 2
    assert all(f"[{k}]" in p.loc[0, "explanation"] for k in p.loc[0, "setups"])
    assert p.loc[0, "setup"] == p.loc[0, "setups"][0]


# ---------------------------------------------------------------- stale / incomplete / missing data
def test_ticker_with_old_latest_bar_is_excluded_not_scanned():
    old = uptrend(idx=IDX[:-3])
    p, m = run({"UP": uptrend(), "OLD": old})
    assert list(p.ticker) == ["UP"] and m["exclusion_reasons_data"] == {"stale_or_missing_latest_bar": 1}


def test_stale_benchmark_refuses_to_produce_picks():
    with pytest.raises(StaleMarketData):
        build_daily_picks({"UP": uptrend()}, spy_frame(IDX[:-5]), ok_reports(["UP"]), S(**PS),
                          expected_bar_date=LAST, earnings_lookup=lambda t: None)


def test_partial_or_future_session_bar_is_not_used():
    extra = pd.bdate_range(LAST + pd.Timedelta(days=1), periods=1)
    spy = pd.concat([spy_frame(), spy_frame().iloc[[-1]].set_axis(extra)])
    f = pd.concat([uptrend(), uptrend().iloc[[-1]].set_axis(extra)])
    p, m = build_daily_picks({"UP": f}, spy, ok_reports(["UP"]), S(**PS), expected_bar_date=LAST,
                             earnings_lookup=lambda t: None)
    assert m["market_bar_date"] == LAST.date().isoformat()
    assert (p["signal_date"] == LAST.date().isoformat()).all() and (p["latest_bar_date"] == m["market_bar_date"]).all()


def test_zero_volume_latest_bar_is_rejected():
    f = uptrend()
    f.iloc[-1, f.columns.get_loc("Volume")] = 0.0
    p, m = run({"Z": f})
    assert p.empty and m["exclusion_reasons_data"] == {"latest_bar_zero_volume": 1}


def test_failed_data_reports_are_counted_by_reason_and_never_picked():
    bad = [DataReport("A", errors=["retrieval failed: boom"]), DataReport("B", errors=["insufficient history: 10 < 260 rows"]),
           DataReport("C", errors=["missing required columns: ['Volume']"]),
           DataReport("D", errors=["incomplete: 40 of 400 benchmark sessions missing"]),
           DataReport("E", errors=["non-positive prices: 1 rows; insufficient history: 1 < 260 rows"])]
    assert [classify_report(r) for r in bad] == ["data_retrieval_failed", "insufficient_history", "invalid_ohlcv",
                                                 "incomplete_vs_benchmark_calendar", "invalid_ohlcv"]
    p, m = build_daily_picks({"UP": uptrend()}, spy_frame(), [*bad, *ok_reports(["UP"])], S(**PS),
                             expected_bar_date=LAST, earnings_lookup=lambda t: pd.Timestamp("2024-12-01"))
    assert m["tickers_requested"] == 6 and m["tickers_excluded_data"] == 5 and list(p.ticker) == ["UP"]
    assert m["exclusion_reasons_data"]["invalid_ohlcv"] == 2


# ---------------------------------------------------------------- earnings / verification
@pytest.mark.parametrize("lookup,status,verified,reason", [
    (lambda t: None, "UNKNOWN", False, "earnings_unverified"),
    (lambda t: pd.Timestamp("2023-02-01"), "UNKNOWN", False, "earnings_unverified"),            # only a past date known
    (lambda t: LAST + pd.Timedelta(days=5), "IN_HOLDING_WINDOW", False, "earnings_in_holding_window"),
    (lambda t: LAST + pd.Timedelta(days=120), "OUTSIDE_HOLDING_WINDOW", True, ""),
])
def test_earnings_status_controls_verification(lookup, status, verified, reason):
    p, _ = run({"UP": uptrend()}, earnings_lookup=lookup)
    r = p.iloc[0]
    assert r["earnings_status"] == status
    assert (r["plan_status"] == "VERIFIED_PLAN") is verified
    assert reason in r["unverified_because"] and (verified or reason)
    if status == "UNKNOWN":
        assert "NOT_CHECKED" in r["event_risk"]


def test_data_failure_or_missing_report_blocks_verification():
    p, _ = build_daily_picks({"UP": uptrend()}, spy_frame(), [], S(**PS), expected_bar_date=LAST,
                             earnings_lookup=lambda t: pd.Timestamp("2024-12-01"))
    assert p.iloc[0]["plan_status"] == "RESEARCH_CANDIDATE" and "data_validation_failed" in p.iloc[0]["unverified_because"]


# ---------------------------------------------------------------- empty list
def test_empty_candidate_list_is_valid_and_complete():
    p, m = run({"FLAT": uptrend(drift=0.0, breakout=False)})
    assert p.empty and m["picks_returned"] == 0 and m["tickers_examined"] == 1 and m["market_bar_date"]
    p2, m2 = run({})
    assert p2.empty and m2["tickers_examined"] == 0


# ---------------------------------------------------------------- price levels
def test_price_levels_and_sizing_from_signal_close_and_atr():
    s = S(initial_capital=1000.0, atr_stop_multiple=2.0, atr_target_multiple=4.0)
    c = pd.DataFrame([{"ticker": "AAA", "setup": "breakout", "setups": ["breakout"], "signal_date": "2024-06-03", "score": 70.0,
                       "score_components": {"trend": 20.0}, "close": 100.0, "sma20": 99.0, "sma50": 95.0, "sma200": 80.0,
                       "ret20": .1, "ret60": .2, "rs20": .05, "rs60": .05, "avg_dollar_volume20": 5e7, "rel_volume": 2.0,
                       "atr14": 2.5, "atr_pct": .025, "_atr": 2.5}])
    r = build_scan_output(c, {"AAA": DataReport("AAA", status="OK", rows=300)}, {"AAA": "2024-06-03"}, s,
                          earnings_lookup=lambda t: pd.Timestamp("2024-12-01"), names={"AAA": "Alpha Corp"}).iloc[0]
    assert (r["stop"], r["target"], r["ref_entry"], r["rr_gross"]) == (95.0, 110.0, 100.0, 2.0)
    assert r["shares"] == 1 and r["position_value"] == 100.0 and r["risk_at_stop_usd"] == 5.0   # 0.5% of $1000 / $5 risk
    assert r["company_name"] == "Alpha Corp" and "not guaranteed" in r["price_label"] and r["rank"] == 1


def test_costs_reduce_net_reward_risk_below_gross():
    s = S(initial_capital=2000.0, commission_bps_per_side=5.0, slippage_bps_per_side=10.0)
    c = pd.DataFrame([{"ticker": "AAA", "setup": "breakout", "setups": ["breakout"], "signal_date": "2024-06-03", "score": 70.0,
                       "score_components": {"trend": 20.0}, "close": 100.0, "sma20": 99.0, "sma50": 95.0, "sma200": 80.0,
                       "ret20": .1, "ret60": .2, "rs20": .05, "rs60": .05, "avg_dollar_volume20": 5e7, "rel_volume": 2.0,
                       "atr14": 2.5, "atr_pct": .025, "_atr": 2.5}])
    r = build_scan_output(c, {}, {"AAA": "2024-06-03"}, s).iloc[0]
    assert 0 < r["rr_net_of_costs"] < r["rr_gross"] and r["est_round_trip_cost_usd"] == pytest.approx(0.3)


# ---------------------------------------------------------------- CLI end-to-end (fake provider, fake clock)
class FakeProvider:
    name = "yfinance"

    def __init__(self, frames):
        self.frames = frames

    def fetch_daily(self, ticker, start, end):
        return self.frames[ticker].copy()


def test_cli_scan_writes_csv_and_json_with_meta(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("swing_screener.cli.next_earnings_date", lambda t: None)
    for mod in ("swing_screener.picks", "swing_screener.data"):
        monkeypatch.setattr(f"{mod}.expected_last_session", lambda as_of=None: LAST)
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[data]\ncache_dir = "{tmp_path / "cache"}"\npause_seconds = 0.0\nmin_history_rows = 50\n'
                   '[universe]\nmin_avg_dollar_volume = 1000000\n[strategy]\nmin_score = 0.0\n[risk]\ninitial_capital = 1000.0\n')
    uni = tmp_path / "u.csv"
    uni.write_text('ticker,exchange,name\nUP,NASDAQ/other,Up Corp Common Stock\n')
    prov = FakeProvider({"SPY": spy_frame(), "UP": uptrend(), "FLAT": uptrend(drift=0.0, breakout=False)})
    out = tmp_path / "out"
    rc = main(["--config", str(cfg), "scan", "--tickers", "UP,FLAT", "--universe-file", str(uni), "--output-dir", str(out)],
              provider=prov)
    assert rc == 0
    data = json.loads((out / "scan.json").read_text())
    assert data["meta"]["market_bar_date"] == LAST.date().isoformat() and data["meta"]["tickers_requested"] == 2
    assert [p["ticker"] for p in data["picks"]] == ["UP"] and data["picks"][0]["company_name"] == "Up Corp Common Stock"
    assert data["picks"][0]["plan_status"] == "RESEARCH_CANDIDATE" and "earnings_unverified" in data["picks"][0]["unverified_because"]
    csv = pd.read_csv(out / "scan.csv")
    assert len(csv) == 1 and ";" in csv.loc[0, "setups"]
    text = capsys.readouterr().out
    assert "market bar date" in text and "Invalidation" in text and "not guaranteed" in text


def test_cli_scan_with_zero_candidates_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("swing_screener.cli.next_earnings_date", lambda t: None)
    for mod in ("swing_screener.picks", "swing_screener.data"):
        monkeypatch.setattr(f"{mod}.expected_last_session", lambda as_of=None: LAST)
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[data]\ncache_dir = "{tmp_path / "cache"}"\npause_seconds = 0.0\nmin_history_rows = 50\n')
    prov = FakeProvider({"SPY": spy_frame(), "FLAT": uptrend(drift=0.0, breakout=False)})
    rc = main(["--config", str(cfg), "scan", "--tickers", "FLAT", "--output-dir", str(tmp_path / "o")], provider=prov)
    assert rc == 0 and json.loads((tmp_path / "o" / "scan.json").read_text())["picks"] == []
    assert "0 candidates" in capsys.readouterr().out


def test_verified_only_keeps_only_fully_verified_and_reranks():
    frames = {"A": uptrend(drift=0.004, jump=0.06), "B": uptrend(drift=0.002), "C": uptrend(drift=0.003)}
    lookup = {"A": LAST + pd.Timedelta(days=3), "B": None, "C": LAST + pd.Timedelta(days=120)}   # in-window / unknown / clear
    p_all, _ = run(frames, earnings_lookup=lambda t: lookup[t])
    assert len(p_all) == 3 and (p_all.plan_status == "VERIFIED_PLAN").sum() == 1
    p, m = run(frames, earnings_lookup=lambda t: lookup[t], verified_only=True)
    assert list(p.ticker) == ["C"] and list(p["rank"]) == [1] and (p.plan_status == "VERIFIED_PLAN").all()
    assert m["qualified_but_unverified_dropped"] == {"earnings_in_holding_window": 1, "earnings_unverified": 1}


def test_verified_only_can_return_zero():
    p, m = run({"A": uptrend()}, earnings_lookup=lambda t: None, verified_only=True)
    assert p.empty and m["verified_only"] and m["tickers_qualified"] == 1
