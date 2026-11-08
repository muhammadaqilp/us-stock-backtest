"""Engine tests on hand-built SYNTHETIC bars and signals. None of this is market data or a market result."""
import pandas as pd
import pytest

from swing_screener.backtest import compute_metrics, simulate
from tests.synthetic import S, flat, frame, signal

ND = 30


def dates_of(df):
    return df.index


def run(prices, sigs, s):
    d = next(iter(prices.values())).index
    return simulate(prices, sigs, d, s)


def px(overrides, n=ND, base=100.0):
    """Flat 100 series with per-index (o,h,l,c) overrides."""
    rows = [(base, base, base, base)] * n
    for i, v in overrides.items():
        rows[i] = v
    return frame(rows)


def test_entry_is_next_session_open_never_same_day():
    a = px({3: (100, 100, 100, 100), 4: (101, 101, 101, 101)})
    d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S())
    t = r.trades.iloc[0]
    assert t.signal_date == d[3].date().isoformat() and t.entry_date == d[4].date().isoformat()
    assert t.entry_price == 101.0   # day-4 OPEN, not day-3 close


def test_no_entry_on_last_session_signal():
    a = flat(10)
    d = a.index
    r = run({"A": a}, {d[-1]: signal(d[-1], ["A"])}, S())
    assert r.trades.empty


def test_entry_and_exit_costs_applied_both_sides():
    s = S(commission_bps_per_side=5, slippage_bps_per_side=10)   # 15 bps/side
    a = px({4: (100, 100, 95, 97)})                               # stop 96 hit on entry-day+? entry day index 4
    d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s)
    t = r.trades.iloc[0]
    assert t.entry_fill == pytest.approx(100 * 1.0015) and t.exit_fill == pytest.approx(96 * 0.9985)
    assert t.pnl == pytest.approx(t.shares * (t.exit_fill - t.entry_fill))
    assert t.commission_usd + t.slippage_usd == pytest.approx((t.entry_notional + t.exit_notional) * 0.0015)


def test_stop_hit_fills_at_stop():
    a = px({5: (100, 101, 95, 97)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_price == 96 and t.exit_date == d[5].date().isoformat()
    assert t.pnl == pytest.approx(125 * -4)


def test_target_hit_fills_at_target():
    a = px({5: (100, 109, 99, 105)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "target" and t.exit_price == 108 and t.pnl == pytest.approx(125 * 8)


def test_gap_through_stop_fills_at_open_not_stop():
    a = px({5: (90, 92, 88, 91)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "stop_gap" and t.exit_price == 90 and t.pnl < 125 * -4


def test_gap_through_target_fills_at_open():
    a = px({5: (112, 113, 111, 112)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "target_gap" and t.exit_price == 112


def test_ambiguous_bar_resolves_to_stop_pessimistically():
    a = px({5: (100, 110, 94, 105)})   # touches both stop (96) and target (108)
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "stop" and bool(t.ambiguous_bar) and t.pnl < 0


def test_same_session_stop_after_entry_is_checked():
    a = px({4: (100, 101, 90, 95)})    # enters at open 100 and the same bar's low breaks the stop
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_date == t.entry_date


def test_max_holding_period_exit_at_close():
    s = S(max_holding_days=5)
    rows = [(100, 100, 100, 100)] * ND
    rows[4:] = [(100, 103, 98, 101)] * (ND - 4)    # stays between stop 96 and target 108
    a = frame(rows); d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s).trades.iloc[0]
    assert t.exit_reason == "time" and t.sessions_held == 5 and t.entry_date == d[4].date().isoformat()
    assert t.exit_date == d[8].date().isoformat() and t.exit_price == 101


def test_end_of_test_liquidation_is_reflected_in_final_equity():
    s = S(commission_bps_per_side=5, slippage_bps_per_side=10)
    rows = [(100, 100, 100, 100)] * 10
    rows[-1] = (100, 103, 99, 102)
    a = frame(rows); d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s)
    t = r.trades.iloc[0]
    last = r.equity.iloc[-1]
    assert t.exit_reason == "end_of_test" and last.open_positions == 0 and last.market_value == 0
    assert last.equity == pytest.approx(s.initial_capital + r.trades.pnl.sum())       # accounting identity
    assert last.equity == pytest.approx(last.cash)
    assert last.equity_pre_liquidation > last.equity                                     # exit cost is in the final number
    m = compute_metrics(r.trades, r.equity, s.initial_capital)
    assert m["total_return_pct"] == pytest.approx((last.equity / s.initial_capital - 1) * 100)


def test_max_concurrent_positions_and_duplicates():
    names = [f"T{i}" for i in range(7)]
    prices = {n: flat(ND) for n in names}
    d = prices["T0"].index
    sigs = {d[3]: signal(d[3], names), d[4]: signal(d[4], names)}   # same names re-signalled while held
    r = run(prices, sigs, S(max_positions=5))
    assert len(r.trades) == 5 and r.trades.ticker.nunique() == 5
    assert (r.equity.open_positions <= 5).all()
    skipped = r.skipped[r.skipped.reason == "max_positions"]
    assert set(skipped.ticker) == {"T5", "T6"}          # lowest-ranked names are the ones dropped
    assert set(r.trades.ticker) == {"T0", "T1", "T2", "T3", "T4"}


def test_one_position_per_ticker_even_if_signalled_repeatedly():
    a = flat(ND); d = a.index
    sigs = {d[i]: signal(d[i], ["A"]) for i in range(3, 10)}
    r = run({"A": a}, sigs, S(max_holding_days=100))
    assert len(r.trades) == 1   # held to end-of-test, never doubled up


def test_cash_limit_respected():
    s = S(initial_capital=10_000.0, risk_per_trade=0.5, max_position_fraction=1.0, max_positions=5)
    prices = {n: flat(ND) for n in "ABC"}
    d = prices["A"].index
    r = run(prices, {d[3]: signal(d[3], list("ABC"), atr=1.0)}, s)
    spent = (r.trades.entry_fill * r.trades.shares).sum()
    assert spent <= 10_000.0 + 1e-6
    assert r.equity.cash.min() >= -1e-9


def test_zero_share_trade_skipped_without_raising_risk():
    s = S(initial_capital=1000.0)
    a = flat(ND, price=100.0); d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"], atr=4.0)}, s)   # risk/share $8 > $5 budget
    assert r.trades.empty and r.skipped.reason.iloc[0].startswith("zero_shares")
    assert r.equity.equity.iloc[-1] == pytest.approx(1000.0)


def test_missing_bar_on_entry_session_skips_and_is_logged():
    a = flat(ND); d = a.index
    b = a.drop(d[4])
    r = simulate({"A": b}, {d[3]: signal(d[3], ["A"])}, d, S())
    assert r.trades.empty and r.skipped.reason.iloc[0] == "no_bar_on_entry_session"


def test_missing_bar_while_holding_carries_last_close_and_counts():
    a = flat(ND); d = a.index
    b = a.drop(d[6])
    r = simulate({"A": b}, {d[3]: signal(d[3], ["A"])}, d, S(max_holding_days=100))
    t = r.trades.iloc[0]
    assert t.missing_bar_days == 1
    assert r.equity.loc[r.equity.date == d[6], "market_value"].iloc[0] == pytest.approx(t.shares * 100)


def test_sizing_uses_current_equity_not_initial():
    # A wins big first; the second position must be sized off the larger equity.
    n = 40
    a = px({5: (100, 110, 100, 108)}, n=n)           # target hit session 5 (+$8/share)
    b = flat(n); d = a.index
    s = S(risk_per_trade=0.01)
    sigs = {d[3]: signal(d[3], ["A"]), d[10]: signal(d[10], ["B"])}
    r = simulate({"A": a, "B": b}, sigs, d, s)
    sh = dict(zip(r.trades.ticker, r.trades.shares))
    assert sh["B"] > sh["A"]


def test_empty_candidates_and_no_trades_metrics_work():
    a = flat(ND)
    r = simulate({"A": a}, {}, a.index, S())
    assert r.trades.empty and (r.equity.equity == 100_000.0).all()
    m = compute_metrics(r.trades, r.equity, 100_000.0)
    assert m["trades"] == 0 and m["total_return_pct"] == 0 and m["profit_factor"] is None


def test_metrics_profit_factor_expectancy_and_drawdown():
    d = pd.bdate_range("2024-01-02", periods=4)
    eq = pd.DataFrame({"date": d, "equity": [100.0, 120.0, 90.0, 110.0], "market_value": [0, 50, 50, 0.0]})
    tr = pd.DataFrame({"pnl": [30.0, -10.0, -10.0], "return_pct": [3.0, -1.0, -1.0], "commission_usd": [1, 1, 1.0],
                       "slippage_usd": [2, 2, 2.0], "entry_notional": [100, 100, 100.0], "exit_notional": [100, 100, 100.0],
                       "sessions_held": [5, 5, 5]})
    m = compute_metrics(tr, eq, 100.0)
    assert m["profit_factor"] == pytest.approx(30 / 20) and m["win_rate_pct"] == pytest.approx(100 / 3)
    assert m["avg_win_usd"] == 30 and m["avg_loss_usd"] == -10 and m["expectancy_usd_after_costs"] == pytest.approx(10 / 3)
    assert m["max_drawdown_pct"] == pytest.approx((90 / 120 - 1) * 100)
    assert m["total_commission_usd"] == 3 and m["turnover_x_avg_equity"] == pytest.approx(600 / 105)


# ---------------------------------------------------------------- optional entry filter (skip big gaps above the signal close)
def _gap_case(open_px, k=0.5, tickers=("A",), max_positions=5):
    bars = {t: px({3: (100, 100, 100, 100), 4: (open_px, open_px, open_px, open_px)}) for t in tickers}
    d = bars[tickers[0]].index
    sig = {d[3]: signal(d[3], list(tickers), atr=2.0, close=100.0)}
    return run(bars, sig, S(max_entry_gap_atr=k, max_positions=max_positions))


def test_entry_filter_allows_open_at_the_limit_and_skips_above_it():
    ok = _gap_case(101.0)                                   # limit = 100 + 0.5 * 2 = 101: allowed (not strictly above)
    assert len(ok.trades) == 1 and ok.trades.iloc[0].entry_price == 101.0
    no = _gap_case(101.5)
    assert len(no.trades) == 0 and list(no.skipped.reason) == ["gap_above_max_entry"]


def test_entry_filter_disabled_by_default_buys_every_open():
    assert len(_gap_case(110.0, k=0.0).trades) == 1


def test_entry_filter_never_blocks_a_gap_down():
    assert len(_gap_case(95.0).trades) == 1


def test_entry_filter_requires_signal_close_instead_of_silently_passing():
    a = px({3: (100, 100, 100, 100), 4: (120, 120, 120, 120)})
    d = a.index
    with pytest.raises(ValueError):
        run({"A": a}, {d[3]: signal(d[3], ["A"])}, S(max_entry_gap_atr=0.5))


def test_entry_filter_skipped_candidate_does_not_block_a_later_one():
    bars = {"A": px({3: (100, 100, 100, 100), 4: (110, 110, 110, 110)}), "B": px({3: (100, 100, 100, 100), 4: (100, 100, 100, 100)})}
    d = bars["A"].index
    r = run(bars, {d[3]: signal(d[3], ["A", "B"], atr=2.0, close=100.0)}, S(max_entry_gap_atr=0.5, max_positions=1))
    assert list(r.trades.ticker) == ["B"] and "gap_above_max_entry" in set(r.skipped.reason)


# ---- breakeven stop (SYNTHETIC). signal(): ATR 2, entry open 100 -> stop 96, target 108, 1-ATR trigger at 102.
def test_breakeven_off_by_default_keeps_original_stop():
    a = px({5: (100, 103, 99, 101), 6: (100, 100, 97, 98), 7: (98, 98, 95, 96)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, S()).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_price == 96 and not t.breakeven_armed


def test_breakeven_raises_stop_from_next_session_only():
    s = S(breakeven_trigger_atr=1.0, commission_bps_per_side=5, slippage_bps_per_side=10)
    be = 100 * 1.0015 / 0.9985
    # session 5 high 103 arms it, and the same bar's low 99 must NOT exit at breakeven; session 6 low 97 does.
    a = px({5: (100, 103, 99, 101), 6: (101, 101, 97, 98)})
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s).trades.iloc[0]
    assert t.exit_reason == "breakeven_stop" and t.exit_date == d[6].date().isoformat()
    assert t.exit_price == pytest.approx(be) and t.stop == 96 and t.exit_stop == pytest.approx(be)
    assert t.pnl == pytest.approx(0, abs=1e-6) and bool(t.breakeven_armed)


def test_breakeven_exit_is_not_counted_as_a_win():
    s = S(breakeven_trigger_atr=1.0, commission_bps_per_side=5, slippage_bps_per_side=10)
    a = px({5: (100, 103, 99, 101), 6: (101, 101, 97, 98)})
    d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s)
    assert compute_metrics(r.trades, r.equity, s.initial_capital)["win_rate_pct"] == 0


def test_breakeven_not_armed_below_trigger_and_gap_uses_open():
    s = S(breakeven_trigger_atr=1.0)
    a = px({5: (100, 101.9, 99, 101), 6: (100, 100, 97, 98)})     # high 101.9 < 102: never armed
    d = a.index
    t = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s).trades.iloc[0]
    assert t.exit_reason == "time" or (t.exit_reason == "stop" and not t.breakeven_armed)
    assert not t.breakeven_armed
    b = px({5: (100, 104, 100, 103), 6: (97, 98, 96.5, 97)})      # armed, then gaps below breakeven
    t = run({"B": b}, {d[3]: signal(d[3], ["B"])}, s).trades.iloc[0]
    assert t.exit_reason == "breakeven_stop_gap" and t.exit_price == 97


def test_breakeven_does_not_block_target_and_ledger_reconciles():
    s = S(breakeven_trigger_atr=1.0)
    a = px({5: (100, 103, 100, 102), 6: (102, 109, 101, 108)})
    d = a.index
    r = run({"A": a}, {d[3]: signal(d[3], ["A"])}, s)
    t = r.trades.iloc[0]
    assert t.exit_reason == "target" and t.exit_price == 108 and bool(t.breakeven_armed)
    assert r.equity.equity.iloc[-1] == pytest.approx(s.initial_capital + r.trades.pnl.sum())
