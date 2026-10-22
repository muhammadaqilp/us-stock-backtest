"""Data layer: provider abstraction, OHLCV validation, metadata, local cache.

Price convention (documented, not hidden):
  * Open/High/Low/Close/Volume are Yahoo's *split-adjusted, NOT dividend-adjusted* values
    (yfinance auto_adjust=False). These approximate historically executable prices and are
    used for signals, fills, stops and sizing.
  * AdjClose is split+dividend adjusted. It is retained for total-return reference only and
    is never used for fills.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

REQUIRED = ("Open", "High", "Low", "Close", "Volume")
ADJUSTMENT = "split_adjusted_not_dividend_adjusted; AdjClose=split+dividend (reference only)"


class DataProviderError(RuntimeError):
    """Raised when a provider cannot return usable data. Never swallowed into fake data."""


class DataProvider(Protocol):
    name: str

    def fetch_daily(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame: ...


class YFinanceProvider:
    """Daily OHLCV via the yfinance Python library (the only supported live source here)."""
    name = "yfinance"

    def fetch_daily(self, ticker, start, end=None):
        import yfinance as yf
        try:
            raw = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                            end=None if end is None else end.strftime("%Y-%m-%d"),
                                            interval="1d", auto_adjust=False, actions=False)
        except Exception as exc:  # network, rate-limit, parsing
            raise DataProviderError(f"{ticker}: yfinance request failed: {exc}") from exc
        if raw is None or raw.empty:
            raise DataProviderError(f"{ticker}: empty response from yfinance")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return normalize_frame(raw)


MCP_ADJUSTMENT = "mcp_fully_adjusted_ohlc_dividend_and_split; NOT executable prices"


class McpExportProvider:
    """Reads a saved yfinance-MCP `get_historical_stock_prices` JSON export (agent-side tool; the CLI cannot call MCP).

    Verified limitation (2026-10-17): the MCP returns Open/High/Low/Close ALREADY dividend-adjusted (auto_adjust=True),
    no raw prices, no start/end parameters (period only). Its frames are tagged with a different adjustment string, so
    the cache will not mix them with executable-price data, and it must not be used for fills/stops/sizing."""
    name = "yfinance-mcp-export"
    adjustment = MCP_ADJUSTMENT

    def __init__(self, files: dict):
        self.files = {k.upper(): v for k, v in files.items()}

    def fetch_daily(self, ticker, start, end=None):
        path = self.files.get(ticker.upper())
        if path is None:
            raise DataProviderError(f"{ticker}: no MCP export file supplied")
        try:
            payload = json.loads(Path(path).read_text())
            rows = json.loads(payload["result"]) if isinstance(payload, dict) and "result" in payload else payload
            raw = pd.DataFrame(rows)
            raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_localize(None).dt.normalize()
            raw = raw.set_index("Date")
        except Exception as exc:
            raise DataProviderError(f"{ticker}: unreadable MCP export {path}: {exc}") from exc
        out = normalize_frame(raw)
        out = out.loc[out.index >= start]
        if end is not None:
            out = out.loc[out.index <= end]
        if out.empty:
            raise DataProviderError(f"{ticker}: MCP export has no rows in requested range")
        return out


def normalize_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename/keep known columns and make the index tz-naive session dates. Does NOT clean or fill values."""
    d = raw.rename(columns={"Adj Close": "AdjClose"}).copy()
    if isinstance(d.index, pd.DatetimeIndex):
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d.index = d.index.normalize()
    keep = [c for c in (*REQUIRED, "AdjClose") if c in d.columns]
    return d[keep]


# ------------------------------------------------------------------ validation
@dataclass
class DataReport:
    ticker: str
    source: str = ""
    retrieved_at: str = ""
    start: str = ""
    end: str = ""
    rows: int = 0
    adjustment: str = ADJUSTMENT
    status: str = "FAILED"          # OK | WARN | FAILED
    stale: bool = False
    lag_sessions: int = 0
    missing_sessions: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "FAILED"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = "; ".join(self.errors)
        d["warnings"] = "; ".join(self.warnings)
        return d


def expected_last_session(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent business day whose daily bar should exist, in US/Eastern.
    Before 16:30 ET the current day's bar is treated as not yet complete. Exchange holidays are
    not modelled; max_stale_sessions provides the tolerance for them."""
    now = pd.Timestamp.now(tz="America/New_York") if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        now = now.tz_localize("America/New_York")
    else:
        now = now.tz_convert("America/New_York")
    day = now.normalize().tz_localize(None)
    if now.hour * 60 + now.minute < 16 * 60 + 30 or day.weekday() >= 5:
        day = day - pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def validate_ohlcv(frame: pd.DataFrame, ticker: str, *, settings=None, calendar: pd.DatetimeIndex | None = None,
                   as_of: pd.Timestamp | None = None, check_stale: bool = True,
                   source: str = "", retrieved_at: str = "") -> DataReport:
    """Validate a daily OHLCV frame. Never modifies data; any hard error => FAILED (ticker excluded)."""
    max_stale = getattr(settings, "max_stale_sessions", 1)
    max_missing = getattr(settings, "max_missing_fraction", 0.02)
    min_rows = getattr(settings, "min_history_rows", 260)
    max_move = getattr(settings, "max_abs_daily_move", 0.50)
    rep = DataReport(ticker=ticker, source=source, retrieved_at=retrieved_at)
    if frame is None or len(frame) == 0:
        rep.errors.append("empty dataframe")
        return rep
    miss = [c for c in REQUIRED if c not in frame.columns]
    if miss:
        rep.errors.append(f"missing required columns: {miss}")
        return rep
    if not isinstance(frame.index, pd.DatetimeIndex):
        rep.errors.append("index is not a DatetimeIndex")
        return rep
    rep.rows = len(frame)
    rep.start, rep.end = frame.index.min().date().isoformat(), frame.index.max().date().isoformat()
    if frame.index.has_duplicates:
        rep.errors.append(f"duplicate dates: {int(frame.index.duplicated().sum())}")
    if not frame.index.is_monotonic_increasing:
        rep.errors.append("dates not sorted ascending")
    ohlcv = frame[list(REQUIRED)]
    if not all(pd.api.types.is_numeric_dtype(ohlcv[c]) for c in REQUIRED):
        rep.errors.append("non-numeric OHLCV values")
        return rep
    nulls = int(ohlcv.isna().any(axis=1).sum())
    if nulls:
        rep.errors.append(f"rows with missing OHLCV values: {nulls}")
    if not np.isfinite(ohlcv.dropna().to_numpy(dtype=float)).all():
        rep.errors.append("non-finite values")
    px = frame[["Open", "High", "Low", "Close"]]
    if (px <= 0).any().any():
        rep.errors.append(f"non-positive prices: {int((px <= 0).any(axis=1).sum())} rows")
    if (frame["Volume"] < 0).any():
        rep.errors.append("negative volume")
    if (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).any() or \
       (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).any():
        rep.errors.append("high/low inconsistent with open/close")
    if len(frame) < min_rows:
        rep.errors.append(f"insufficient history: {len(frame)} < {min_rows} rows")
    if not rep.errors:
        zero_vol = float((frame["Volume"] == 0).mean())
        if zero_vol > 0.01:
            rep.warnings.append(f"zero-volume days: {zero_vol:.1%}")
        moves = frame["Close"].pct_change().abs()
        if (moves > max_move).any():
            rep.warnings.append(f"{int((moves > max_move).sum())} daily moves > {max_move:.0%} (possible bad tick / unadjusted split)")
    if calendar is not None and rep.rows and not rep.errors:
        window = calendar[(calendar >= frame.index.min()) & (calendar <= frame.index.max())]
        missing = window.difference(frame.index)
        rep.missing_sessions = len(missing)
        if len(window) and len(missing) / len(window) > max_missing:
            rep.errors.append(f"incomplete: {len(missing)} of {len(window)} benchmark sessions missing")
        elif len(missing):
            rep.warnings.append(f"{len(missing)} benchmark sessions missing (not filled)")
    if check_stale and rep.rows and not rep.errors:
        exp = expected_last_session(as_of)
        last = frame.index.max()
        rep.lag_sessions = max(0, int(np.busday_count(last.date(), exp.date())))
        if rep.lag_sessions > max_stale:
            rep.stale = True
            rep.warnings.append(f"STALE: last bar {rep.end}, expected ~{exp.date()} (lag {rep.lag_sessions} sessions)")
    rep.status = "FAILED" if rep.errors else ("WARN" if rep.warnings else "OK")
    return rep


def drop_incomplete_last_bar(frame: pd.DataFrame):
    """Yahoo sometimes returns the newest session as an unfinished row (e.g. Close missing). Only the LAST row may be
    dropped, and only for missing OHLCV; the drop is returned as a warning so it is never silent. Missing values
    anywhere else are left in place and still fail validation."""
    if frame is None or len(frame) < 2 or not all(c in frame.columns for c in REQUIRED):
        return frame, ""
    last = frame.iloc[-1]
    missing = [c for c in REQUIRED if pd.isna(last[c])]
    if not missing:
        return frame, ""
    return frame.iloc[:-1], (f"INCOMPLETE LATEST BAR DROPPED: {frame.index[-1].date()} had missing {', '.join(missing)} "
                             f"(Yahoo unfinished row); newest usable bar is {frame.index[-2].date()}")


# ------------------------------------------------------------------ cache + retrieval
@dataclass
class History:
    ticker: str
    frame: pd.DataFrame
    report: DataReport


def _paths(cache_dir, ticker):
    base = Path(cache_dir) / ticker.replace("/", "_")
    return base.with_suffix(".csv"), base.with_suffix(".meta.json")


def _read_cache(cache_dir, ticker):
    csv, meta = _paths(cache_dir, ticker)
    if not (csv.exists() and meta.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
        frame = pd.read_csv(csv, index_col=0, parse_dates=True)
        return frame, m
    except Exception:
        return None  # unreadable cache is treated as a miss, not as data


def _write_cache(cache_dir, ticker, frame, meta):
    csv, mp = _paths(cache_dir, ticker)
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv)
    mp.write_text(json.dumps(meta, indent=2))


def get_history(ticker: str, provider: DataProvider, settings, *, start: pd.Timestamp, end: pd.Timestamp | None = None,
                refresh: bool = False, max_cache_age_hours: float | None = None,
                calendar: pd.DatetimeIndex | None = None, as_of: pd.Timestamp | None = None,
                check_stale: bool = True) -> History:
    """Return validated history. Raises DataProviderError if retrieval fails and no valid cache covers the request.
    Cache is used unless refresh=True, the cache was built with a different provider/adjustment, does not cover
    `start`, or is older than max_cache_age_hours."""
    cached = None if refresh else _read_cache(settings.cache_dir, ticker)
    use = False
    if cached is not None:
        frame, m = cached
        age_ok = True
        if max_cache_age_hours is not None:
            age = (datetime.now(UTC) - datetime.fromisoformat(m["retrieved_at"])).total_seconds() / 3600
            age_ok = age <= max_cache_age_hours
        covers = len(frame) > 0 and frame.index.min() <= start + pd.Timedelta(days=7)
        use = (m.get("source") == provider.name and m.get("adjustment") == getattr(provider, "adjustment", ADJUSTMENT) and age_ok and covers
               and (end is None or frame.index.max() >= end - pd.Timedelta(days=7)))
    if use:
        src, retrieved = m["source"] + " (cache)", m["retrieved_at"]
    else:
        frame = provider.fetch_daily(ticker, start, end)  # may raise DataProviderError
        retrieved = datetime.now(UTC).isoformat(timespec="seconds")
        src = provider.name
        _write_cache(settings.cache_dir, ticker, frame,
                     {"source": provider.name, "retrieved_at": retrieved, "adjustment": getattr(provider, "adjustment", ADJUSTMENT),
                      "start": start.date().isoformat(), "rows": len(frame)})
    frame = frame.loc[frame.index >= start] if len(frame) else frame
    if end is not None and len(frame):
        frame = frame.loc[frame.index <= end]
    frame, dropped_note = drop_incomplete_last_bar(frame)
    rep = validate_ohlcv(frame, ticker, settings=settings, calendar=calendar, as_of=as_of,
                         check_stale=check_stale, source=src, retrieved_at=retrieved)
    if dropped_note:
        rep.warnings.append(dropped_note)
        if rep.status == "OK":
            rep.status = "WARN"
    rep.adjustment = getattr(provider, "adjustment", ADJUSTMENT)
    if rep.adjustment != ADJUSTMENT:
        rep.warnings.append("NOT EXECUTABLE PRICES: dividend-adjusted OHLC; unsuitable for fills/stops/sizing")
        if rep.status == "OK":
            rep.status = "WARN"
    return History(ticker, frame, rep)


def load_universe_history(tickers, provider, settings, *, start, end=None, refresh=False,
                          max_cache_age_hours=None, calendar=None, as_of=None, check_stale=True):
    """Fetch many tickers. Per-ticker failures are recorded, never fabricated. Returns (frames, reports)."""
    frames, reports = {}, []
    for t in tickers:
        try:
            h = get_history(t, provider, settings, start=start, end=end, refresh=refresh,
                            max_cache_age_hours=max_cache_age_hours, calendar=calendar, as_of=as_of,
                            check_stale=check_stale)
            reports.append(h.report)
            if h.report.ok:
                frames[t] = h.frame
            fetched = not h.report.source.endswith("(cache)")
        except DataProviderError as exc:
            r = DataReport(ticker=t, source=provider.name, errors=[f"retrieval failed: {exc}"])
            reports.append(r)
            fetched = True
        if fetched and settings.pause_seconds and getattr(provider, "name", "") == "yfinance":
            time.sleep(settings.pause_seconds)
    return frames, reports


def reports_to_frame(reports) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in reports])
