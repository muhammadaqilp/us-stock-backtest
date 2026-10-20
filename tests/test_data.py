"""Data-layer tests. All frames are SYNTHETIC."""
import json

import pandas as pd
import pytest

from swing_screener.data import (
    ADJUSTMENT,
    DataProviderError,
    expected_last_session,
    get_history,
    load_universe_history,
    normalize_frame,
    validate_ohlcv,
)
from tests.synthetic import S, random_walk

AS_OF = pd.Timestamp("2024-12-31 18:00", tz="America/New_York")


def good(n=300, start="2024-01-02"):
    return random_walk(n, 1, start=start)


def rep(df, **kw):
    kw.setdefault("check_stale", False)
    return validate_ohlcv(df, "TEST", settings=S(), **kw)


def test_valid_data_ok():
    r = rep(good())
    assert r.status == "OK" and r.rows == 300 and r.errors == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.drop(columns=["Volume"]), "missing required columns"),
    (lambda d: d.assign(Close=d.Close.where(d.index != d.index[5])), "missing OHLCV"),
    (lambda d: d.assign(Open=d.Open.where(d.index != d.index[5], -1.0)), "non-positive"),
    (lambda d: d.assign(High=d.High.where(d.index != d.index[5], d.Low.iloc[5] * 0.5)), "inconsistent"),
    (lambda d: d.assign(Volume=d.Volume.where(d.index != d.index[5], -10.0)), "negative volume"),
    (lambda d: d.iloc[:20], "insufficient history"),
    (lambda d: d.iloc[0:0], "empty"),
])
def test_malformed_data_fails(mutate, needle):
    r = rep(mutate(good()))
    assert r.status == "FAILED" and any(needle in e for e in r.errors), r.errors


def test_duplicate_and_unsorted_dates_fail():
    d = good()
    dup = pd.concat([d, d.iloc[[3]]]).sort_index()
    assert any("duplicate" in e for e in rep(dup).errors)
    assert any("not sorted" in e for e in rep(d.iloc[::-1]).errors)


def test_missing_bars_vs_calendar_flagged_not_filled():
    d = good()
    holes = d.drop(d.index[[10, 11, 12, 13, 14, 15, 16, 17]])  # 8/300 = 2.7% missing -> FAILED
    r = rep(holes, calendar=d.index)
    assert r.status == "FAILED" and r.missing_sessions == 8
    assert len(holes) == 292  # nothing was filled
    small = rep(d.drop(d.index[[10, 11]]), calendar=d.index)
    assert small.status == "WARN" and small.missing_sessions == 2


def test_stale_data_detected_and_not_treated_as_current():
    d = good(300, start="2024-01-02")  # ends 2025-02-24 -> use as_of far later
    late = validate_ohlcv(d, "T", settings=S(), as_of=pd.Timestamp("2025-03-20 18:00", tz="America/New_York"))
    assert late.stale and late.status == "WARN" and "STALE" in late.warnings[0]
    fresh = validate_ohlcv(d, "T", settings=S(), as_of=pd.Timestamp("2025-02-24 18:00", tz="America/New_York"))
    assert not fresh.stale


def test_expected_last_session_rules():
    # Wednesday 12:00 ET: today's bar not complete -> Tuesday
    assert expected_last_session(pd.Timestamp("2024-10-16 12:00", tz="America/New_York")).date().isoformat() == "2024-10-15"
    assert expected_last_session(pd.Timestamp("2024-10-16 17:00", tz="America/New_York")).date().isoformat() == "2024-10-16"
    # Saturday -> Friday
    assert expected_last_session(pd.Timestamp("2024-10-19 10:00", tz="America/New_York")).date().isoformat() == "2024-10-18"


def test_normalize_strips_timezone_and_renames_adjclose():
    d = good(5)
    d.index = d.index.tz_localize("America/New_York")
    d["Adj Close"] = d["Close"]
    n = normalize_frame(d)
    assert n.index.tz is None and "AdjClose" in n.columns


class FakeProvider:
    name = "yfinance"

    def __init__(self, frames=None, fail=()):
        self.frames, self.fail, self.calls = frames or {}, set(fail), 0

    def fetch_daily(self, ticker, start, end):
        self.calls += 1
        from swing_screener.data import DataProviderError
        if ticker in self.fail or ticker not in self.frames:
            raise DataProviderError(f"{ticker}: simulated provider failure")
        return self.frames[ticker]


def test_failed_retrieval_raises_and_is_recorded_per_ticker(tmp_path):
    s = S(cache_dir=str(tmp_path))
    p = FakeProvider({"AAA": good()}, fail={"BBB"})
    with pytest.raises(DataProviderError):
        get_history("BBB", p, s, start=pd.Timestamp("2024-01-01"))
    frames, reports = load_universe_history(["AAA", "BBB", "CCC"], p, s, start=pd.Timestamp("2024-01-01"), check_stale=False)
    assert list(frames) == ["AAA"]
    status = {r.ticker: r.status for r in reports}
    assert status == {"AAA": "OK", "BBB": "FAILED", "CCC": "FAILED"}
    assert "retrieval failed" in reports[1].errors[0]


def test_cache_used_then_refresh_forces_refetch_and_metadata_recorded(tmp_path):
    s = S(cache_dir=str(tmp_path))
    p = FakeProvider({"AAA": good()})
    start = pd.Timestamp("2024-01-01")
    h1 = get_history("AAA", p, s, start=start, check_stale=False)
    h2 = get_history("AAA", p, s, start=start, check_stale=False)
    assert p.calls == 1 and h2.report.source.endswith("(cache)")
    get_history("AAA", p, s, start=start, refresh=True, check_stale=False)
    assert p.calls == 2
    meta = json.loads((tmp_path / "AAA.meta.json").read_text())
    assert meta["adjustment"] == ADJUSTMENT and meta["source"] == "yfinance" and meta["retrieved_at"]
    assert h1.report.retrieved_at and h1.report.rows == 300 and h1.report.start and h1.report.end


def test_cache_with_other_adjustment_convention_is_not_reused(tmp_path):
    s = S(cache_dir=str(tmp_path))
    p = FakeProvider({"AAA": good()})
    start = pd.Timestamp("2024-01-01")
    get_history("AAA", p, s, start=start, check_stale=False)
    mp = tmp_path / "AAA.meta.json"
    m = json.loads(mp.read_text()); m["adjustment"] = "auto_adjust=True"; mp.write_text(json.dumps(m))
    get_history("AAA", p, s, start=start, check_stale=False)
    assert p.calls == 2


def test_cache_older_than_max_age_is_refetched(tmp_path):
    s = S(cache_dir=str(tmp_path))
    p = FakeProvider({"AAA": good()})
    start = pd.Timestamp("2024-01-01")
    get_history("AAA", p, s, start=start, check_stale=False)
    mp = tmp_path / "AAA.meta.json"
    m = json.loads(mp.read_text()); m["retrieved_at"] = "2020-01-01T00:00:00+00:00"; mp.write_text(json.dumps(m))
    get_history("AAA", p, s, start=start, max_cache_age_hours=12, check_stale=False)
    assert p.calls == 2
