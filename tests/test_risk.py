"""Sizing and cost arithmetic (SYNTHETIC numbers)."""
import pytest

from swing_screener.risk import buy_fill, levels, sell_fill, size_position
from tests.synthetic import S


def test_fills_apply_costs_on_both_sides():
    assert buy_fill(100, 0.0015) == pytest.approx(100.15)
    assert sell_fill(100, 0.0015) == pytest.approx(99.85)


def test_levels_and_reward_risk():
    s = S(commission_bps_per_side=5, slippage_bps_per_side=10)
    lv = levels(100.0, 2.0, s)
    assert lv["stop"] == 96 and lv["target"] == 108 and lv["rr_gross"] == 2.0
    assert lv["rr_net"] < 2.0   # costs reduce reward/risk
    assert not levels(100.0, 60.0, s)["valid"]  # stop would be <= 0


def test_size_risk_limited():
    s = S()  # equity 100k, risk 0.5% = 500, rps = 4 -> 125 shares (cap 200, cash ok)
    assert size_position(100_000, 100_000, 100.0, 96.0, s)["shares"] == 125


def test_size_position_cap_limited():
    s = S()  # tight stop -> risk would allow 1000 shares; cap 20% = 200
    assert size_position(100_000, 100_000, 100.0, 99.5, s)["shares"] == 200


def test_size_cash_limited():
    s = S()
    assert size_position(100_000, 5_000, 100.0, 96.0, s)["shares"] == 50


def test_zero_shares_never_escalates_risk():
    s = S(initial_capital=1000.0)  # risk budget $5, risk/share $4 on $100 stock -> 1 share; with $8 risk/share -> 0
    z = size_position(1000.0, 1000.0, 100.0, 92.0, s)
    assert z["shares"] == 0 and z["reason"] == "risk_budget"
    c = size_position(1000.0, 1000.0, 300.0, 296.0, s)   # cap 20% = $200 < $300 price
    assert c["shares"] == 0 and c["reason"] == "position_cap"


def test_sizing_includes_costs_in_risk_per_share():
    s0, s1 = S(), S(commission_bps_per_side=50, slippage_bps_per_side=50)
    assert size_position(100_000, 100_000, 100.0, 96.0, s1)["shares"] < size_position(100_000, 100_000, 100.0, 96.0, s0)["shares"]
