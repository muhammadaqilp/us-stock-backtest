"""Indicator, qualification and scoring tests (SYNTHETIC inputs)."""
import numpy as np
import pandas as pd
import pytest

from swing_screener.indicators import add_indicators, atr_wilder
from swing_screener.scoring import WEIGHTS, score_row
from swing_screener.screener import build_scan_output, rank_candidates
from swing_screener.strategies import (
    established_uptrend,
    passes_common_filters,
    qualify_all,
    qualify_breakout,
    qualify_pullback,
    qualify_relative_strength,
)
from tests.synthetic import S, frame, random_walk


def test_sma_values():
    c = np.arange(1.0, 251.0)
    d = frame([(x, x + 1, x - 0.5, x) for x in c])
    ind = add_indicators(d, S())
    assert ind["sma20"].iloc[19] == pytest.approx(np.mean(c[:20]))
    assert ind["sma50"].iloc[-1] == pytest.approx(np.mean(c[-50:]))
    assert ind["sma200"].iloc[-1] == pytest.approx(np.mean(c[-200:]))
    assert np.isnan(ind["sma200"].iloc[198])


def test_prior_high_excludes_current_session():
    rows = [(10, 11, 9, 10)] * 30 + [(10, 50, 9, 10)]  # last bar spikes high to 50
    ind = add_indicators(frame(rows), S())
    assert ind["prior_high"].iloc[-1] == 11   # NOT 50: today's high is not in today's threshold
    assert ind["prior_high"].iloc[-2] == 11


def test_indicators_are_causal_truncation_invariant():
    d = random_walk(400, 3)
    spy = random_walk(400, 4)
    full = add_indicators(d, S(), spy)
    cut = 300
    part = add_indicators(d.iloc[:cut], S(), spy.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-12, atol=1e-12)


def test_atr_wilder_constant_range():
    n = 60
    h = pd.Series(11.0, index=range(n)); l = pd.Series(9.0, index=range(n)); c = pd.Series(10.0, index=range(n))
    a = atr_wilder(h, l, c)
    assert np.isnan(a.iloc[12]) and a.iloc[-1] == pytest.approx(2.0)


def test_relative_volume_uses_prior_baseline():
    rows = [(10, 11, 9, 10)] * 25
    d = frame(rows)
    d.iloc[-1, d.columns.get_loc("Volume")] = 3_000_000
    ind = add_indicators(d, S())
    assert ind["rel_volume"].iloc[-1] == pytest.approx(3.0)   # 3M / 1M baseline (spike excluded from its own baseline)


def test_missing_benchmark_bar_gives_nan_not_ffill():
    d = random_walk(120, 5); spy = random_walk(120, 6).drop(random_walk(120, 6).index[100])
    ind = add_indicators(d, S(), spy)
    assert np.isnan(ind["rs60"].iloc[100]) and np.isfinite(ind["rs60"].iloc[99])


# ---- qualification on hand-built rows (SYNTHETIC)
def row(**kw):
    base = dict(Open=100.0, High=104.0, Low=99.0, Close=102.0, sma20=100.0, sma50=95.0, sma200=80.0, pb_ma=100.0,
                sma50_slope20=0.03, prior_high=101.0, ret20=0.08, ret60=0.15, rs20=0.03, rs60=0.05, avg_dollar_vol20=5e7,
                rel_volume=2.0, atr14=2.0, atr_pct=0.02, recent_high=106.0, pb_touch_ratio=1.0, pb_depth=0.05,
                high_rs_window=104.0, prev_high=101.5, prev_close=100.5)
    base.update(kw)
    return pd.Series(base, name=pd.Timestamp("2024-06-03"))


def test_uptrend_definition():
    assert established_uptrend(row())
    assert not established_uptrend(row(Close=94.0))        # below SMA50
    assert not established_uptrend(row(sma50=79.0))        # SMA50 below SMA200


def test_breakout_rules():
    s = S()
    assert qualify_breakout(row(), s)[0]
    assert not qualify_breakout(row(Close=101.0), s)[0]            # equal to prior high is not a break
    assert not qualify_breakout(row(rel_volume=1.2), s)[0]         # volume below threshold
    assert not qualify_breakout(row(sma50=79.0), s)[0]             # no uptrend


def test_pullback_requires_reversal_not_just_proximity():
    s = S()
    ok = row(Close=102.0, Open=100.0, prev_high=101.5, pb_ma=100.0)
    assert qualify_pullback(ok, s)[0]
    near_but_no_reversal = row(Close=99.5, Open=100.5, prev_high=101.5, pb_ma=100.0)   # bearish bar, below MA
    assert not qualify_pullback(near_but_no_reversal, s)[0]
    no_higher_high = row(Close=101.0, Open=100.0, prev_high=101.5)                     # bullish but not above prior high
    assert not qualify_pullback(no_higher_high, s)[0]
    no_touch = row(pb_touch_ratio=1.05)
    assert not qualify_pullback(no_touch, s)[0]
    too_shallow = row(pb_depth=0.01)
    assert not qualify_pullback(too_shallow, s)[0]


def test_relative_strength_rules():
    s = S()
    assert qualify_relative_strength(row(), s)[0]
    assert not qualify_relative_strength(row(rs60=-0.01), s)[0]       # underperforms on one lookback
    assert not qualify_relative_strength(row(ret20=-0.01), s)[0]      # no momentum
    assert not qualify_relative_strength(row(high_rs_window=130.0), s)[0]  # too far from high
    assert not qualify_relative_strength(row(rs20=np.nan), s)[0]      # missing data never passes


def test_common_filters():
    s = S()
    assert passes_common_filters(row(), s)[0]
    assert passes_common_filters(row(Close=4.0), s) == (False, "price_below_min")
    assert passes_common_filters(row(avg_dollar_vol20=1e6), s) == (False, "illiquid")
    assert passes_common_filters(row(atr_pct=0.2), s) == (False, "volatility_filter")
    assert passes_common_filters(row(atr14=np.nan), s) == (False, "incomplete_indicators")
    assert qualify_all(row(Close=4.0), s) == []


def test_score_reproducible_bounded_and_weights():
    s = S()
    r = row()
    a, ca = score_row(r, s); b, cb = score_row(r.copy(), s)
    assert a == b and ca == cb
    assert sum(WEIGHTS.values()) == 100 and 0 <= a <= 100
    assert all(0 <= ca[k] <= WEIGHTS[k] for k in WEIGHTS)
    worst = row(sma50_slope20=-1, ret20=-1, ret60=-1, rs20=-1, rs60=-1, rel_volume=0.1, atr_pct=0.5)
    assert score_row(worst, s)[0] < a
    best = row(sma50_slope20=1, ret20=1, ret60=1, rs20=1, rs60=1, rel_volume=9, atr_pct=0.001)
    assert score_row(best, s)[0] == pytest.approx(100.0)


def test_qualification_is_independent_of_score_threshold():
    s_lo, s_hi = S(min_score=0), S(min_score=99)
    assert qualify_all(row(), s_lo) == qualify_all(row(), s_hi) != []


def test_rank_candidates_empty_and_deterministic():
    s = S()
    assert rank_candidates({}, s).empty
    flat = add_indicators(frame([(10, 10.5, 9.5, 10)] * 300), s)   # no uptrend -> nothing
    assert rank_candidates({"X": flat}, s).empty
    d = random_walk(500, 11, drift=0.003)
    ind = {"AAA": add_indicators(d, S(min_avg_dollar_volume=0, min_score=0), random_walk(500, 12))}
    s0 = S(min_avg_dollar_volume=0, min_score=0)
    r1, r2 = rank_candidates(ind, s0, as_of=ind["AAA"].index[-1]), rank_candidates(ind, s0, as_of=ind["AAA"].index[-1])
    pd.testing.assert_frame_equal(r1, r2)


def test_scan_output_distinguishes_candidate_from_plan():
    from swing_screener.data import DataReport
    s = S(initial_capital=1000.0, min_avg_dollar_volume=0, min_score=0)
    cands = pd.DataFrame([{"ticker": "AAA", "setup": "breakout", "signal_date": "2024-06-03", "score": 70.0,
                           "score_components": {"trend": 20.0}, "close": 50.0, "sma20": 49.0, "sma50": 45.0, "sma200": 40.0,
                           "ret20": .1, "ret60": .2, "rs20": .05, "rs60": .05, "avg_dollar_volume20": 5e7, "rel_volume": 2.0,
                           "atr14": 1.0, "atr_pct": .02, "_atr": 1.0}])
    ok = DataReport("AAA", status="OK", rows=300)
    out = build_scan_output(cands, {"AAA": ok}, {"AAA": "2024-06-03"}, s, earnings_lookup=lambda t: pd.Timestamp("2024-09-01"))
    assert out.loc[0, "plan_status"] == "VERIFIED_PLAN" and out.loc[0, "shares"] >= 1
    stale = DataReport("AAA", status="WARN", stale=True)
    out2 = build_scan_output(cands, {"AAA": stale}, {"AAA": "2024-06-03"}, s)
    assert out2.loc[0, "plan_status"] == "RESEARCH_CANDIDATE" and "stale_data" in out2.loc[0, "unverified_because"]
    pricey = cands.assign(close=900.0, atr14=20.0, _atr=20.0)
    out3 = build_scan_output(pricey, {"AAA": ok}, {"AAA": "2024-06-03"}, s)
    assert out3.loc[0, "plan_status"] == "RESEARCH_CANDIDATE" and "zero_shares" in out3.loc[0, "unverified_because"]
