"""`plan` command tests. All data is SYNTHETIC."""
import json

import pandas as pd
import pytest

from swing_screener.cli import main
from swing_screener.data import DataProviderError, DataReport
from swing_screener.indicators import add_indicators
from swing_screener.picks import StaleMarketData, build_daily_picks
from swing_screener.plan import build_plan, checklist, plan_steps, setups_from_checklist
from swing_screener.strategies import qualify_all
from tests.synthetic import S, random_walk
from tests.test_picks import IDX, LAST, PS, ok_reports, spy_frame, uptrend

CLEAR = lambda t: LAST + pd.Timedelta(days=120)


def plan(frame, s=None, ticker="UP", report="ok", **kw):
    s = s or S(**PS)
    rep = DataReport(ticker, status="OK", rows=300) if report == "ok" else report
    kw.setdefault("earnings_lookup", CLEAR)
    return build_plan(ticker, frame, spy_frame(), rep, s, expected_bar_date=LAST, **kw)


def test_checklist_agrees_with_the_real_qualifiers_on_many_synthetic_bars():
    s = S(min_avg_dollar_volume=0, min_price=1.0, breakout_volume_multiple=1.0, general_relative_volume=0.5)
    checked = hit_any = 0
    for seed in range(6):
        f = random_walk(700, seed, start="2021-01-04", drift=0.001, vol=0.02)
        spy = random_walk(700, 100 + seed, start="2021-01-04", drift=0.0004, vol=0.008, price=300)
        ind = add_indicators(f, s, spy)
        for i in range(260, 700, 3):
            r = ind.iloc[i]
            assert setups_from_checklist(checklist(r, s)) == qualify_all(r, s), (seed, i)
            checked += 1
            hit_any += bool(qualify_all(r, s))
    assert checked > 700 and hit_any > 10          # the comparison included real positives, not just empty lists


def test_qualified_plan_matches_the_scan_row_exactly():
    s = S(**PS)
    frames = {"UP": uptrend()}
    picks, _ = build_daily_picks(frames, spy_frame(), ok_reports(frames), s, expected_bar_date=LAST, earnings_lookup=CLEAR)
    p = plan(frames["UP"])
    row = picks.iloc[0]
    assert p["status"] == "SETUP_QUALIFIED" and p["plan_status"] == "VERIFIED_PLAN" and p["unverified_because"] == ""
    for k in ("score", "stop", "target", "shares", "ref_entry", "rr_gross", "rr_net_of_costs", "risk_at_stop_usd"):
        assert p[k] == pytest.approx(row[k]), k
    assert p["setups"] == list(row["setups"]) and p["levels_are"] == "PLAN_REFERENCE_LEVELS"
    assert plan_steps(p, s)[0].startswith("Reference entry")


def test_no_setup_ticker_lists_failed_rules_and_labels_levels_hypothetical():
    p = plan(uptrend(drift=0.0, breakout=False))
    assert p["status"] == "NO_SETUP_TODAY" and p["setups"] == [] and p["plan_status"] == "NO_TRADE_SETUP"
    assert any(not c["passed"] for k in ("breakout", "pullback", "relative_strength") for c in p["checklist"][k])
    steps = plan_steps(p, S(**PS)) if "stop" in p else []
    assert all(x.startswith(("HYPOTHETICAL", "Time exit", "Size", "Earnings")) for x in steps)


def test_setup_met_but_score_below_minimum_is_not_a_signal():
    p = plan(uptrend(), S(**{**PS, "min_score": 99.9}))
    assert p["setups"] and p["status"] == "SETUP_BELOW_MIN_SCORE" and p["plan_status"] != "VERIFIED_PLAN"
    assert "NOT A SIGNAL" in plan_steps(p, S(**PS))[0]


@pytest.mark.parametrize("lookup,reason", [(lambda t: None, "earnings_unverified"),
                                           (lambda t: LAST + pd.Timedelta(days=4), "earnings_in_holding_window")])
def test_earnings_unknown_or_in_window_blocks_verification(lookup, reason):
    p = plan(uptrend(), earnings_lookup=lookup)
    assert p["status"] == "SETUP_QUALIFIED" and p["plan_status"] == "RESEARCH_CANDIDATE" and reason in p["unverified_because"]


def test_zero_share_plan_explains_and_never_raises_risk():
    p = plan(uptrend(price=500.0))                       # one share exceeds the $5 risk budget / 20% cap
    assert p["shares"] == 0 and "risk is never raised" in p["sizing_note"] and "zero_shares" in p["unverified_because"]


def test_unavailable_or_not_current_data_gives_no_plan():
    bad = DataReport("UP", errors=["insufficient history: 10 < 260 rows"])
    assert plan(uptrend(), report=bad)["status"] == "DATA_UNAVAILABLE"
    assert plan(uptrend(), report=None)["status"] == "DATA_UNAVAILABLE"
    old = plan(uptrend(idx=IDX[:-3]))
    assert old["status"] == "DATA_NOT_CURRENT" and "stop" not in old
    z = uptrend()
    z.iloc[-1, z.columns.get_loc("Volume")] = 0.0
    assert plan(z)["status"] == "DATA_NOT_CURRENT"


def test_stale_benchmark_refuses_a_plan():
    with pytest.raises(StaleMarketData):
        build_plan("UP", uptrend(), spy_frame(IDX[:-5]), DataReport("UP", status="OK"), S(**PS), expected_bar_date=LAST)


def test_incomplete_indicators_fail_checklist_not_pass():
    short = uptrend().iloc[-120:]                         # < 200 sessions: SMA200 undefined
    p = plan(short)
    assert p["status"] == "NO_SETUP_TODAY" and not p["checklist"]["common"][0]["passed"] and "stop" not in p


class FakeProvider:
    name = "yfinance"

    def __init__(self, frames):
        self.frames = frames

    def fetch_daily(self, ticker, start, end):
        if ticker not in self.frames:
            raise DataProviderError(f"{ticker}: empty response from yfinance")
        return self.frames[ticker].copy()


def _cfg(tmp_path):
    c = tmp_path / "c.toml"
    c.write_text(f'[data]\ncache_dir = "{tmp_path / "cache"}"\npause_seconds = 0.0\nmin_history_rows = 50\n'
                 '[universe]\nmin_avg_dollar_volume = 1000000\n[strategy]\nmin_score = 0.0\n[risk]\ninitial_capital = 1000.0\n')
    return str(c)


def _patch(monkeypatch):
    monkeypatch.setattr("swing_screener.cli.next_earnings_date", lambda t: None)
    for mod in ("swing_screener.picks", "swing_screener.data"):
        monkeypatch.setattr(f"{mod}.expected_last_session", lambda as_of=None: LAST)


def test_cli_plan_accepts_lowercase_and_mixed_inputs_and_writes_files(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch)
    uni = tmp_path / "u.csv"
    uni.write_text("ticker,exchange,name\nUP,NASDAQ/other,Up Corp Common Stock\n")
    prov = FakeProvider({"SPY": spy_frame(), "UP": uptrend(), "FLAT": uptrend(drift=0.0, breakout=False)})
    out = tmp_path / "o"
    rc = main(["--config", _cfg(tmp_path), "plan", "up,flat", "NOPE", "--universe-file", str(uni), "--output-dir", str(out)],
              provider=prov)
    assert rc == 0
    text = capsys.readouterr().out
    assert "Up Corp Common Stock" in text and "NO_SETUP_TODAY" in text and "DATA_UNAVAILABLE" in text
    assert "not guaranteed" in text and "Checks (common + uptrend)" in text
    data = json.loads((out / "plan.json").read_text())["plans"]
    assert [p["ticker"] for p in data] == ["FLAT", "NOPE", "UP"] and data[2]["status"] == "SETUP_QUALIFIED"
    assert len(pd.read_csv(out / "plan.csv")) == 3


def test_cli_plan_exits_nonzero_on_outage(tmp_path, capsys):
    from tests.test_walkforward import DownProvider
    rc = main(["--config", _cfg(tmp_path), "plan", "AAA", "--output-dir", str(tmp_path / "o")], provider=DownProvider())
    assert rc == 2 and "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------- news (display only)
def test_news_parser_handles_both_layouts_and_never_scores():
    from swing_screener.events import recent_headlines
    nested = [{"content": {"title": "Co beats", "provider": {"displayName": "Wire"}, "pubDate": "2026-10-15T10:00:00Z",
                           "canonicalUrl": {"url": "http://x/1"}}}]
    flat = [{"title": "Co cuts guidance", "publisher": "Pub", "providerPublishTime": 1_760_000_000, "link": "http://x/2"}]
    a = recent_headlines("UP", fetch=lambda t: nested)
    b = recent_headlines("UP", fetch=lambda t: flat)
    assert a["status"] == b["status"] == "OK" and a["headlines"][0]["publisher"] == "Wire"
    assert b["headlines"][0]["title"] == "Co cuts guidance" and b["headlines"][0]["published"].startswith("2025")
    assert len(recent_headlines("UP", n=1, fetch=lambda t: nested + nested)["headlines"]) == 1


@pytest.mark.parametrize("fetch", [lambda t: [], lambda t: None, lambda t: (_ for _ in ()).throw(RuntimeError("boom")),
                                   lambda t: [{"no_title": 1}, "junk"]])
def test_empty_or_failed_news_is_not_available_never_no_news(fetch):
    from swing_screener.events import recent_headlines
    n = recent_headlines("UP", fetch=fetch)
    assert n["status"] == "NOT_AVAILABLE" and n["headlines"] == [] and n["link"].endswith("/UP/news")
    assert "NOT" in n["note"].upper() or "failed" in n["note"]


def test_news_does_not_change_score_ranking_or_verification():
    from swing_screener.events import recent_headlines
    a = plan(uptrend())
    b = plan(uptrend(), news_lookup=lambda t: recent_headlines(t, fetch=lambda x: [{"title": "Bad news"}]))
    assert b["news"]["status"] == "OK" and a["news"]["status"] == "NOT_CHECKED"
    for k in ("score", "stop", "target", "shares", "plan_status", "status"):
        assert a[k] == b[k]
    frames = {"UP": uptrend()}
    p, _ = build_daily_picks(frames, spy_frame(), ok_reports(frames), S(**PS), expected_bar_date=LAST, earnings_lookup=CLEAR,
                             news_lookup=lambda t: recent_headlines(t, fetch=lambda x: []))
    assert p.iloc[0]["news_status"] == "NOT_AVAILABLE" and "/UP/news" in p.iloc[0]["news_link"]


def test_max_entry_price_shown_only_when_filter_enabled():
    off = plan(uptrend())
    on = plan(uptrend(), S(**{**PS, "max_entry_gap_atr": 0.5}))
    assert off["max_entry_price"] == "" and not any("Entry filter" in x for x in plan_steps(off, S(**PS)))
    assert on["max_entry_price"] == pytest.approx(on["ref_entry"] + 0.5 * on["atr14"], abs=1e-3)
    assert any("skip the trade if the open is above" in x for x in plan_steps(on, S(**{**PS, "max_entry_gap_atr": 0.5})))


# ---------------------------------------------------------------- plan --entry (user-supplied fill price)
def test_plan_at_entry_arithmetic_matches_atr_multiples():
    from swing_screener.plan import plan_at_entry
    s = S(**{**PS, "initial_capital": 100_000.0})
    p = plan(uptrend(), s)
    atr, ref = p["atr14"], p["ref_entry"]
    a = plan_at_entry(p, ref + 1.0, s)
    assert a["available"] and a["entry"] == pytest.approx(ref + 1.0, abs=1e-4)
    assert a["stop"] == pytest.approx(ref + 1.0 - 2 * atr, abs=1e-3) and a["target"] == pytest.approx(ref + 1.0 + 4 * atr, abs=1e-3)
    assert a["rr_gross"] == 2.0 and a["gap_vs_reference_pct"] == pytest.approx(100 / ref, abs=0.01)
    assert a["shares"] >= 1 and a["risk_at_stop_usd"] <= 100_000 * s.risk_per_trade + 1e-9   # risk budget respected


def test_plan_at_entry_same_price_as_reference_matches_the_plan_levels():
    from swing_screener.plan import plan_at_entry
    s = S(**PS)
    p = plan(uptrend(), s)
    a = plan_at_entry(p, p["ref_entry"], s)
    for k in ("stop", "target", "shares", "rr_gross"):
        assert a[k] == pytest.approx(p[k], abs=1e-3)


def test_plan_at_entry_zero_shares_explains_and_never_raises_risk():
    from swing_screener.plan import plan_at_entry
    s = S(**PS)                                          # $1,000 capital
    p = plan(uptrend(price=500.0), s)
    a = plan_at_entry(p, p["ref_entry"], s)
    assert a["shares"] == 0 and "risk is never raised" in a["sizing_note"] and "risk budget" in a["sizing_note"]


@pytest.mark.parametrize("mult,needle", [(1.6, "typo, wrong currency"), (1.0001, None)])
def test_plan_at_entry_warns_on_implausible_entry(mult, needle):
    from swing_screener.plan import plan_at_entry
    s = S(**PS)
    p = plan(uptrend(), s)
    w = " ".join(plan_at_entry(p, p["ref_entry"] * mult, s)["warnings"])
    assert (needle in w) if needle else ("typo" not in w)


def test_plan_at_entry_flags_gap_filter_unqualified_setup_and_earnings():
    from swing_screener.plan import plan_at_entry
    s = S(**{**PS, "max_entry_gap_atr": 0.5})
    p = plan(uptrend(), s, earnings_lookup=lambda t: None)
    w = " ".join(plan_at_entry(p, p["ref_entry"] + 3 * p["atr14"], s)["warnings"])
    assert "above the optional max entry price" in w and "large gap up" in w and "earnings status UNKNOWN" in w
    q = plan(uptrend(drift=0.0, breakout=False), s)
    assert q["status"] == "NO_SETUP_TODAY" and "ref_entry" in q
    assert "not a qualified signal" in " ".join(plan_at_entry(q, q["ref_entry"], s)["warnings"])


def test_plan_at_entry_unavailable_without_levels_or_with_impossible_stop():
    from swing_screener.plan import plan_at_entry
    s = S(**PS)
    assert not plan_at_entry(plan(uptrend(idx=IDX[:-3])), 100.0, s)["available"]     # DATA_NOT_CURRENT: no ATR
    p = plan(uptrend(), s)
    assert not plan_at_entry(p, 0.5 * p["atr14"], s)["available"]                    # stop would be <= 0


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "nan", "inf"])
def test_cli_rejects_invalid_entry_values(tmp_path, bad, capsys):
    with pytest.raises(SystemExit):
        main(["--config", _cfg(tmp_path), "plan", "UP", "--entry", bad], provider=None)
    err = capsys.readouterr().err
    assert "positive number" in err or "is not a number" in err


def test_cli_entry_requires_exactly_one_ticker(tmp_path, capsys):
    rc = main(["--config", _cfg(tmp_path), "plan", "AAA", "BBB", "--entry", "10"], provider=None)
    assert rc == 2 and "exactly one ticker" in capsys.readouterr().err


def test_cli_plan_with_entry_prints_and_saves_the_entry_plan(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch)
    prov = FakeProvider({"SPY": spy_frame(), "UP": uptrend()})
    out = tmp_path / "o"
    ref = round(float(uptrend()["Close"].iloc[-1]), 2)
    rc = main(["--config", _cfg(tmp_path), "plan", "UP", "--entry", str(ref + 1), "--output-dir", str(out)], provider=prov)
    assert rc == 0
    text = capsys.readouterr().out
    assert "Plan at YOUR entry" in text and "price you supplied" in text
    data = json.loads((out / "plan.json").read_text())["plans"][0]
    assert data["at_entry"]["available"] and data["at_entry"]["entry"] == pytest.approx(ref + 1)
    row = pd.read_csv(out / "plan.csv").iloc[0]
    assert row["at_entry_stop"] == pytest.approx(data["at_entry"]["stop"]) and row["at_entry_shares"] == data["at_entry"]["shares"]


def test_plan_reports_a_data_note_when_the_newest_valid_bar_lags():
    extra = pd.bdate_range(LAST + pd.Timedelta(days=1), periods=1)[0]
    s = S(**PS)
    p = build_plan("UP", uptrend(), spy_frame(), DataReport("UP", status="OK", rows=300), s, expected_bar_date=extra,
                   earnings_lookup=CLEAR)
    assert p["status"] == "SETUP_QUALIFIED" and "1 session(s) behind" in p["data_note"] and p["market_bar_date"] == LAST.date().isoformat()
    assert plan(uptrend())["data_note"] == ""
