"""Walk-forward structure, OOS separation, sensitivity, universe and CLI failure handling (all SYNTHETIC data)."""
import json

import pandas as pd
import pytest

from swing_screener.backtest import (
    InsufficientHistory,
    make_windows,
    run_walk_forward,
    spy_buy_and_hold,
)
from swing_screener.cli import main
from swing_screener.data import DataProviderError
from swing_screener.universe import filter_listings, sample_universe, snapshot
from tests.synthetic import S, random_walk

# Relaxed thresholds so the synthetic walk produces trades; this is about plumbing, not strategy quality.
SW = dict(min_avg_dollar_volume=0, min_score=0, min_price=1.0, breakout_volume_multiple=1.0, general_relative_volume=0.0,
          initial_capital=10_000.0, commission_bps_per_side=5.0, slippage_bps_per_side=10.0)


def universe(n_days=1500, tickers=6):
    spy = random_walk(n_days, 900, drift=0.0004, vol=0.008, price=300)
    return {f"S{i}": random_walk(n_days, i, drift=0.0012, vol=0.02) for i in range(tickers)}, spy


def test_windows_are_sequential_non_overlapping_after_warmup():
    cal = pd.bdate_range("2019-01-02", periods=1500)
    s = S()
    w = make_windows(cal, s)
    assert w[0][0] >= cal[0] + pd.DateOffset(years=2)
    for (a0, b0, _), (a1, b1, _) in zip(w, w[1:]):
        assert a1 > b0 >= a0           # no overlap, strictly increasing
        assert (a1 - b0).days <= 5            # contiguous (weekend gap only)
    assert not w[0][2]                         # first window is a full window


def test_insufficient_history_raises():
    cal = pd.bdate_range("2023-01-02", periods=400)   # < 2y warmup + 3 months
    with pytest.raises(InsufficientHistory):
        make_windows(cal, S())
    with pytest.raises(InsufficientHistory):
        run_walk_forward({}, random_walk(400, 1), S())


def test_walkforward_oos_only_and_reports_windows():
    prices, spy = universe()
    s = S(**SW)
    res = run_walk_forward(prices, spy, s)
    first_oos = res["oos_period"]["start"]
    assert first_oos >= (spy.index[0] + pd.DateOffset(years=2)).date().isoformat()
    assert res["strategy_combined_oos"]["trades"] > 0, "fixture must trade or the OOS tests below are vacuous"
    assert len(res["windows"]) == res["oos_period"]["n_windows"] >= 2
    wins = res["windows"]
    assert all(w["window_start"] > p["window_end"] for p, w in zip(wins, wins[1:]))
    # window returns chain to the combined return
    growth = 1.0
    for w in wins:
        growth *= 1 + w["total_return_pct"] / 100
    assert (growth - 1) * 100 == pytest.approx(res["strategy_combined_oos"]["total_return_pct"], abs=1e-6)


def test_no_future_leakage_changing_later_data_does_not_alter_earlier_results(tmp_path):
    prices, spy = universe()
    s = S(**SW)
    base = run_walk_forward(prices, spy, s, output_dir=tmp_path / "a")
    cutoff = spy.index[-260]
    altered = {t: f.copy() for t, f in prices.items()}
    for f in altered.values():
        later = f.index >= cutoff
        f.loc[later, ["Open", "High", "Low", "Close"]] *= 0.5     # crash the future
    mod = run_walk_forward(altered, spy, s, output_dir=tmp_path / "b")
    ta = pd.read_csv(tmp_path / "a" / "trades.csv"); tb = pd.read_csv(tmp_path / "b" / "trades.csv")
    early_a = ta[ta.exit_date < cutoff.date().isoformat()].reset_index(drop=True)
    early_b = tb[tb.exit_date < cutoff.date().isoformat()].reset_index(drop=True)
    assert len(early_a) > 0
    pd.testing.assert_frame_equal(early_a, early_b)
    wa, wb = base["windows"], mod["windows"]
    assert wa[0] == wb[0]            # first (oldest) window unchanged by what happens later


def test_cost_sensitivity_monotonic_and_reports_written(tmp_path):
    prices, spy = universe()
    res = run_walk_forward(prices, spy, S(**SW), output_dir=tmp_path)
    sens = res["cost_sensitivity"]
    ends = [r["end_equity"] for r in sens]
    assert [r["cost_multiplier"] for r in sens] == [1.0, 2.0, 3.0]
    assert ends[0] >= ends[1] >= ends[2] and ends[0] > ends[2]
    for name in ("summary.json", "trades.csv", "equity.csv", "windows.csv", "cost_sensitivity.csv", "skipped_entries.csv"):
        assert (tmp_path / name).exists()
    j = json.loads((tmp_path / "summary.json").read_text())
    assert j["status"] == "RESEARCH_ONLY_NOT_VALIDATED" and "Survivorship" in j["caveats"][0]
    c = j["strategy_combined_oos"]
    for k in ("total_return_pct", "annualized_return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "avg_win_usd",
              "avg_loss_usd", "profit_factor", "expectancy_usd_after_costs", "exposure_pct", "turnover_x_avg_equity",
              "total_commission_usd", "total_slippage_usd"):
        assert k in c
    assert c["end_equity"] == pytest.approx(10_000 + pd.read_csv(tmp_path / "trades.csv").pnl.sum())


def test_zero_candidates_walkforward_is_valid():
    prices, spy = universe()
    res = run_walk_forward(prices, spy, S(**{**SW, "min_price": 1e9}))      # nothing passes the price filter
    c = res["strategy_combined_oos"]
    assert c["trades"] == 0 and c["total_return_pct"] == 0 and c["exposure_pct"] == 0


def test_spy_buy_and_hold_arithmetic():
    spy = random_walk(100, 2, price=100)
    s = S(commission_bps_per_side=10, slippage_bps_per_side=0, initial_capital=1000.0)
    b = spy_buy_and_hold(spy, spy.index[10:60], s)
    o, c = spy.Open.iloc[10], spy.Close.iloc[59]
    expected = 1000 / (o * 1.001) * c * 0.999
    assert b["end_equity"] == pytest.approx(expected)


# ---- universe + provider failure paths
def test_universe_filter_removes_etfs_tests_and_non_common():
    a = pd.DataFrame({"Symbol": ["AAPL", "QQQ", "TEST", "WXYZW", "GOOD"], "ETF": ["N", "Y", "N", "N", "N"],
                      "Test Issue": ["N", "N", "Y", "N", "N"], "Financial Status": ["N"] * 5,
                      "Security Name": ["Apple Inc. Common Stock", "Invesco QQQ ETF", "Test", "Foo Acquisition Corp Warrant",
                                        "Good Co Common Stock"]})
    b = pd.DataFrame({"ACT Symbol": ["IBM", "BRK.B"], "ETF": ["N", "N"], "Test Issue": ["N", "N"], "Exchange": ["N", "N"],
                      "Security Name": ["IBM Common Stock", "Berkshire Hathaway Class B"]})
    out = filter_listings(a, b)
    assert list(out.ticker) == ["AAPL", "BRK-B", "GOOD", "IBM"]
    with pytest.raises(ValueError):
        filter_listings(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]}))


def test_sample_is_deterministic_and_snapshot_hash_stable(tmp_path):
    tk = [f"T{i:03d}" for i in range(200)]
    assert sample_universe(tk, 20, 42) == sample_universe(tk, 20, 42) != sample_universe(tk, 20, 7)
    i1 = snapshot(tk[:5], tk[:3], tk[3:5], tmp_path / "x"); i2 = snapshot(list(reversed(tk[:5])), tk[:3], [], tmp_path / "y")
    assert i1["sha256_16"] == i2["sha256_16"] and (tmp_path / "x" / "universe_snapshot.csv").exists()


class DownProvider:
    name = "yfinance"

    def fetch_daily(self, ticker, start, end):
        raise DataProviderError(f"{ticker}: simulated outage")


def _cfg(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(f'[data]\ncache_dir = "{tmp_path / "cache"}"\npause_seconds = 0.0\n')
    return str(p)


@pytest.mark.parametrize("cmd", ["data", "scan", "backtest"])
def test_cli_reports_provider_outage_and_exits_nonzero(tmp_path, capsys, cmd):
    rc = main(["--config", _cfg(tmp_path), cmd, "--tickers", "AAA", "--output-dir", str(tmp_path / "out")], provider=DownProvider())
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err
    assert not (tmp_path / "out" / "scan.csv").exists()
