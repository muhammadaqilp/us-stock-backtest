"""Tests for MCP-export adapter, event risk, and ledger reconciliation. All data is SYNTHETIC."""
import json

import pandas as pd
import pytest

from swing_screener.backtest import SimResult, assert_reconciled, run_walk_forward
from swing_screener.data import (
    ADJUSTMENT,
    MCP_ADJUSTMENT,
    DataProviderError,
    McpExportProvider,
    get_history,
)
from swing_screener.events import NOT_CHECKED, event_risk_text
from tests.synthetic import S, random_walk
from tests.test_walkforward import SW, universe


def _mcp_file(tmp_path, df, name="X.json"):
    rows = [{"Date": f"{d.date()}T04:00:00.000Z", "Open": r.Open, "High": r.High, "Low": r.Low, "Close": r.Close,
             "Volume": int(r.Volume), "Dividends": 0.0, "Stock Splits": 0.0, "Capital Gains": 0.0}
            for d, r in df.iterrows()]
    p = tmp_path / name
    p.write_text(json.dumps({"result": json.dumps(rows)}))   # MCP wraps a JSON string in {"result": ...}
    return p


def test_mcp_export_parses_and_is_tagged_not_executable(tmp_path):
    df = random_walk(300, 1, start="2024-01-02")
    prov = McpExportProvider({"X": _mcp_file(tmp_path, df)})
    s = S(cache_dir=str(tmp_path / "c"))
    h = get_history("X", prov, s, start=df.index[0], check_stale=False)
    assert len(h.frame) == 300 and h.report.adjustment == MCP_ADJUSTMENT != ADJUSTMENT
    assert h.report.status == "WARN" and any("NOT EXECUTABLE" in w for w in h.report.warnings)


def test_mcp_export_missing_or_corrupt_raises(tmp_path):
    with pytest.raises(DataProviderError):
        McpExportProvider({}).fetch_daily("X", pd.Timestamp("2024-01-01"), None)
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(DataProviderError):
        McpExportProvider({"X": bad}).fetch_daily("X", pd.Timestamp("2024-01-01"), None)


def test_mcp_cache_not_reused_by_executable_provider(tmp_path):
    df = random_walk(300, 1, start="2024-01-02")
    s = S(cache_dir=str(tmp_path / "c"))
    get_history("X", McpExportProvider({"X": _mcp_file(tmp_path, df)}), s, start=df.index[0], check_stale=False)

    class Exec:
        name = "yfinance-mcp-export"   # same name, different adjustment convention
        calls = 0

        def fetch_daily(self, t, a, b):
            Exec.calls += 1
            return df
    get_history("X", Exec(), s, start=df.index[0], check_stale=False)
    assert Exec.calls == 1


def test_event_risk_never_guesses():
    d = pd.Timestamp("2024-06-03")
    assert event_risk_text(None, d, 15) == NOT_CHECKED
    assert event_risk_text(pd.Timestamp("2024-06-10"), d, 15).startswith("EARNINGS_IN_HOLDING_WINDOW")
    assert "outside" in event_risk_text(pd.Timestamp("2024-09-01"), d, 15)
    assert NOT_CHECKED in event_risk_text(pd.Timestamp("2024-05-01"), d, 15)


def test_reconciliation_guard_detects_broken_ledger():
    eq = pd.DataFrame({"equity": [1000.0, 1050.0], "open_positions": [0, 0]})
    ok = SimResult(pd.DataFrame({"pnl": [50.0]}), eq, pd.DataFrame())
    assert_reconciled(ok, 1000.0)
    with pytest.raises(AssertionError):
        assert_reconciled(SimResult(pd.DataFrame({"pnl": [10.0]}), eq, pd.DataFrame()), 1000.0)
    open_pos = eq.assign(open_positions=[0, 1])
    with pytest.raises(AssertionError):
        assert_reconciled(SimResult(ok.trades, open_pos, pd.DataFrame()), 1000.0)


def test_walkforward_reports_by_setup_and_reconciles():
    prices, spy = universe()
    res = run_walk_forward(prices, spy, S(**SW))
    assert isinstance(res["by_setup"], dict)
    assert sum(v["trades"] for v in res["by_setup"].values()) == res["strategy_combined_oos"]["trades"]


def test_earnings_in_window_downgrades_plan_to_research_candidate():
    from swing_screener.data import DataReport
    from swing_screener.screener import build_scan_output
    s = S(initial_capital=1000.0, min_avg_dollar_volume=0, min_score=0)
    cands = pd.DataFrame([{"ticker": "AAA", "setup": "breakout", "signal_date": "2024-06-03", "score": 70.0,
                           "score_components": {"trend": 20.0}, "close": 50.0, "sma20": 49.0, "sma50": 45.0, "sma200": 40.0,
                           "ret20": .1, "ret60": .2, "rs20": .05, "rs60": .05, "avg_dollar_volume20": 5e7, "rel_volume": 2.0,
                           "atr14": 1.0, "atr_pct": .02, "_atr": 1.0}])
    rep = {"AAA": DataReport("AAA", status="OK", rows=300)}
    out = build_scan_output(cands, rep, {"AAA": "2024-06-03"}, s, earnings_lookup=lambda t: pd.Timestamp("2024-06-10"))
    assert out.loc[0, "plan_status"] == "RESEARCH_CANDIDATE" and "earnings_in_holding_window" in out.loc[0, "unverified_because"]
