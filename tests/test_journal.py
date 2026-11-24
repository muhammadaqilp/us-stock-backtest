"""Paper-trading journal tests. All frames are SYNTHETIC; nothing here is a market result."""
import json

import numpy as np
import pandas as pd
import pytest

from swing_screener.backtest import simulate
from swing_screener.cli import main
from swing_screener.journal import (
    append_to_journal,
    compute_outcomes,
    follow_pick,
    journal_rows,
    load_journal,
    summarize,
)
from tests.synthetic import S, frame, random_walk
from tests.test_picks import LAST, FakeProvider, spy_frame, uptrend


def row(signal_date, atr=2.0, close=100.0, **kw):
    base = {"signal_date": pd.Timestamp(signal_date).date().isoformat(), "ticker": "A", "signal_close": close, "atr14": atr,
            "atr_stop_multiple": 2.0, "atr_target_multiple": 4.0, "max_holding_days": 15, "max_entry_gap_atr": 0.0,
            "breakeven_trigger_atr": 0.0, "cost_rate": 0.0}
    return pd.Series({**base, **kw})


def flat_px(overrides, n=30, base=100.0):
    rows = [(base, base, base, base)] * n
    for i, v in overrides.items():
        rows[i] = v
    return frame(rows)


# ---------------------------------------------------------------- same exits as the backtest
@pytest.mark.parametrize("kw", [{}, {"breakeven_trigger_atr": 1.0}, {"max_entry_gap_atr": 0.25},
                                {"commission_bps_per_side": 5.0, "slippage_bps_per_side": 10.0, "breakeven_trigger_atr": 1.0}])
def test_journal_scores_every_trade_exactly_like_the_backtest(kw):
    s = S(initial_capital=1e9, max_positions=1, **kw)
    compared, reasons = 0, set()
    for seed in range(40):
        f = random_walk(45, seed, start="2024-01-02", drift=0.0, vol=0.03, price=100.0)
        d = f.index
        atr = float(f["Close"].iloc[5]) * 0.02
        sig = pd.DataFrame([{"ticker": "A", "setup": "breakout", "score": 80.0, "signal_date": d[5].date().isoformat(),
                             "_atr": atr, "close": float(f["Close"].iloc[5])}])
        bt = simulate({"A": f}, {d[5]: sig}, d, s)
        j = follow_pick(row(d[5], atr=atr, close=float(f["Close"].iloc[5]), max_holding_days=s.max_holding_days,
                            max_entry_gap_atr=s.max_entry_gap_atr, breakeven_trigger_atr=s.breakeven_trigger_atr,
                            cost_rate=s.cost_rate), f, d)
        if bt.trades.empty:
            assert j["status"] == "skipped_gap_above_max_entry", seed
            continue
        t = bt.trades.iloc[0]
        if t.exit_reason.startswith("end_of_test"):
            assert j["status"] == "open"
            continue
        assert (j["exit_reason"], j["exit_date"], j["entry_date"]) == (t.exit_reason, t.exit_date, t.entry_date), seed
        assert j["exit_price"] == pytest.approx(t.exit_price) and j["return_pct"] == pytest.approx(t.return_pct)
        compared += 1
        reasons.add(t.exit_reason)
    assert compared >= 25 and "target" in reasons and any(r.endswith("stop") for r in reasons)
    if s.breakeven_trigger_atr == 0:
        assert "time" in reasons
    else:
        assert any(r.startswith("breakeven") for r in reasons)      # the breakeven path was really exercised


def test_awaiting_entry_no_entry_bar_and_gap_skip():
    f = flat_px({})
    d = f.index
    assert follow_pick(row(d[-1]), f, d)["status"] == "awaiting_entry"
    assert follow_pick(row(d[3]), f.drop(d[4]), d)["status"] == "no_entry_bar"      # missing bar is never filled
    g = flat_px({4: (101, 101, 101, 101)})
    out = follow_pick(row(d[3], max_entry_gap_atr=0.25), g, d)                   # open 101 > 100 + 0.25*2
    assert out["status"] == "skipped_gap_above_max_entry" and np.isnan(out["r_multiple"])


def test_r_multiple_time_exit_and_open_trade_mark():
    f = flat_px({5: (100, 105, 99, 104)})
    d = f.index
    out = follow_pick(row(d[3], max_holding_days=3), f, d)
    assert out["exit_reason"] == "time" and out["exit_date"] == d[6].date().isoformat() and out["sessions_held"] == 3
    assert out["r_multiple"] == pytest.approx(0.0)                                  # closed back at 100: 0R
    short = f.iloc[:7]                                                              # data ends before any exit
    o = follow_pick(row(d[3]), short, short.index)
    assert o["status"] == "open" and o["exit_reason"] == "" and o["r_multiple"] == pytest.approx(0.0)
    t = follow_pick(row(d[3]), flat_px({5: (100, 109, 99, 105)}), d)
    assert t["exit_reason"] == "target" and t["r_multiple"] == pytest.approx(2.0)   # +8 on 4 risk


def test_breakeven_exit_is_not_a_win_and_payoff_math():
    out = pd.DataFrame({
        "signal_date": ["2024-01-02"] * 5, "rank": [1, 2, 3, 4, 5], "in_top_n": [True] * 5, "top_n": [10] * 5,
        "plan_status": ["VERIFIED_PLAN"] * 5, "setups": ["breakout"] * 5, "status": ["closed"] * 4 + ["open"],
        "r_multiple": [2.0, 1.0, -1.0, 1e-12, 0.5], "return_pct": [4.0, 2.0, -2.0, 0.0, 1.0]})
    g = {x["group"]: x for x in summarize(out)}["scan_top_n"]
    assert g["closed"] == 4 and g["open"] == 1
    assert g["win_rate_pct"] == pytest.approx(50.0)                                  # the ~0R exit is a loss, not a win
    assert g["avg_win_r"] == pytest.approx(1.5) and g["avg_loss_r"] == pytest.approx(-0.5)
    assert g["payoff_ratio"] == pytest.approx(3.0) and g["breakeven_win_rate_pct"] == pytest.approx(25.0)
    assert g["profit_factor"] == pytest.approx(3.0) and g["expectancy_r"] == pytest.approx(0.5)
    assert g["enough_trades"] is False


def test_groups_rerank_within_each_day():
    out = pd.DataFrame({
        "signal_date": ["d1"] * 4 + ["d2"] * 2, "rank": [1, 2, 3, 4, 1, 2], "top_n": [2] * 6,
        "in_top_n": [True, True, False, False, True, True],
        "plan_status": ["RESEARCH_CANDIDATE", "VERIFIED_PLAN", "VERIFIED_PLAN", "VERIFIED_PLAN", "VERIFIED_PLAN", "X"],
        "setups": ["relative_strength", "breakout", "pullback;relative_strength", "breakout", "relative_strength", "pullback"],
        "status": ["closed"] * 6, "r_multiple": [1.0] * 6, "return_pct": [1.0] * 6})
    g = {x["group"]: x for x in summarize(out)}
    assert g["scan_top_n"]["closed"] == 4 and g["all_qualified"]["closed"] == 6
    assert g["verified_top_n"]["closed"] == 3                 # d1: ranks 2,3 (top 2 of verified); d2: rank 1
    assert g["no_relative_strength_top_n"]["closed"] == 3     # d1: ranks 2,3; d2: rank 2


# ---------------------------------------------------------------- recording
def _picks(signal_date="2024-03-01"):
    return pd.DataFrame([{"rank": i, "ticker": t, "setups": ["breakout"], "score": 90 - i, "plan_status": "RESEARCH_CANDIDATE",
                          "unverified_because": "earnings_unverified", "earnings_status": "UNKNOWN", "signal_date": signal_date,
                          "close": 50.0, "atr14": 1.0} for i, t in enumerate(["AAA", "BBB", "CCC"], start=1)])


def test_rows_store_their_own_rules_and_dates_are_never_overwritten(tmp_path):
    s = S(top_n=2, atr_target_multiple=3.0, commission_bps_per_side=5, slippage_bps_per_side=10)
    rows = journal_rows(_picks(), s, "2024-03-01T22:00:00+00:00")
    assert list(rows["in_top_n"]) == [True, True, False]
    assert (rows["atr_target_multiple"] == 3.0).all() and rows["cost_rate"].iloc[0] == pytest.approx(0.0015)
    n, _ = append_to_journal(tmp_path, rows, "2024-03-01")
    assert n == 3
    again = journal_rows(_picks().assign(score=1), s, "2024-03-01T23:00:00+00:00")
    n2, msg = append_to_journal(tmp_path, again, "2024-03-01")
    assert n2 == 0 and "already" in msg
    j = load_journal(tmp_path)
    assert len(j) == 3 and list(j["score"]) == [89, 88, 87] and j["in_top_n"].dtype == bool
    assert append_to_journal(tmp_path, journal_rows(_picks("2024-03-04"), s, "x"), "2024-03-04")[0] == 3
    assert len(load_journal(tmp_path)) == 6


def test_outcomes_use_recorded_rules_not_current_config():
    f = flat_px({5: (100, 107, 99, 105)})
    d = f.index
    j = pd.DataFrame([row(d[3], atr_target_multiple=3.0), row(d[3], atr_target_multiple=4.0)])
    out = compute_outcomes(j, {"A": f}, d)
    assert list(out["exit_reason"]) == ["target", "time"]
    assert out["target"].tolist() == [106.0, 108.0]


# ---------------------------------------------------------------- CLI end-to-end (fake provider, fake clock)
def _cfg(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[data]\ncache_dir = "{tmp_path / "cache"}"\npause_seconds = 0.0\nmin_history_rows = 50\n'
                   '[universe]\nmin_avg_dollar_volume = 1000000\n[strategy]\nmin_score = 0.0\n[risk]\ninitial_capital = 1000.0\n')
    return str(cfg)


def _clock(monkeypatch, day):
    for mod in ("swing_screener.picks", "swing_screener.data", "swing_screener.cli"):
        monkeypatch.setattr(f"{mod}.expected_last_session", lambda as_of=None: day)


def test_cli_scan_journal_then_journal_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("swing_screener.cli.next_earnings_date", lambda t: None)
    _clock(monkeypatch, LAST)
    up = uptrend()
    prov = FakeProvider({"SPY": spy_frame(), "UP": up, "FLAT": uptrend(drift=0.0, breakout=False)})
    jdir = tmp_path / "journal"
    args = ["--config", _cfg(tmp_path), "scan", "--tickers", "UP,FLAT", "--output-dir", str(tmp_path / "o"), "--no-news"]
    assert main(args + ["--journal", str(jdir)], provider=prov) == 0
    assert main(args + ["--journal", str(jdir)], provider=prov) == 0               # same day again: not duplicated
    text = capsys.readouterr().out
    assert "recorded 1 candidate" in text and "already has signal date" in text
    j = load_journal(jdir)
    assert list(j["ticker"]) == ["UP"] and j["signal_date"].iloc[0] == LAST.date().isoformat()

    # three more sessions arrive: entry at the next open, then a big up day through the target
    more = pd.bdate_range(LAST + pd.Timedelta(days=1), periods=3)
    c0, atr = float(up["Close"].iloc[-1]), float(j["atr14"].iloc[0])
    ext = pd.DataFrame({"Open": [c0, c0, c0 + 5 * atr], "High": [c0 + 0.5, c0 + 0.5, c0 + 5 * atr],
                        "Low": [c0 - 0.5, c0 - 0.5, c0 + 4 * atr], "Close": [c0, c0, c0 + 5 * atr], "Volume": 5e6}, index=more)
    spy2 = spy_frame(up.index.append(more))
    prov2 = FakeProvider({"SPY": spy2, "UP": pd.concat([up, ext])})
    _clock(monkeypatch, more[-1])
    assert main(["--config", _cfg(tmp_path), "journal", "--dir", str(jdir), "--refresh"], provider=prov2) == 0
    out = pd.read_csv(jdir / "outcomes.csv")
    assert out.loc[0, "status"] == "closed" and out.loc[0, "exit_reason"] == "target_gap"
    assert out.loc[0, "entry_date"] == more[0].date().isoformat()
    rep = json.loads((jdir / "report.json").read_text())
    top = {g["group"]: g for g in rep["groups"]}["scan_top_n"]
    assert top["closed"] == 1 and top["win_rate_pct"] == 100.0 and rep["as_of_session"] == more[-1].date().isoformat()
    assert "PAPER-TRADING JOURNAL" in capsys.readouterr().out


def test_cli_journal_empty_is_clean(tmp_path, capsys):
    assert main(["--config", _cfg(tmp_path), "journal", "--dir", str(tmp_path / "none")], provider=None) == 0
    assert "is empty" in capsys.readouterr().out
