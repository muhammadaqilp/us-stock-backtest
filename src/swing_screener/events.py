"""Forward-looking event risk for SCAN output only (not usable in historical backtests: yfinance has no point-in-time
earnings history). Any failure or missing field yields NOT_CHECKED, never a guess."""
from __future__ import annotations

import pandas as pd

NOT_CHECKED = "NOT_CHECKED: earnings date unavailable; verify earnings/news independently before trading"


def next_earnings_date(ticker: str):
    """Next earnings date from yfinance Ticker.calendar, or None if unavailable."""
    import yfinance as yf
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None
        return pd.Timestamp(min(dates)).normalize()
    except Exception:
        return None


def event_risk_text(earnings: pd.Timestamp | None, signal_date: pd.Timestamp, max_holding_days: int) -> str:
    """Flag earnings falling inside the maximum holding window (calendar approximation: 1.5 days per session)."""
    if earnings is None:
        return NOT_CHECKED
    horizon = signal_date + pd.Timedelta(days=int(max_holding_days * 1.5) + 1)
    if earnings < signal_date:
        return f"last-known earnings date {earnings.date()} is in the past; next date unknown. {NOT_CHECKED}"
    if earnings <= horizon:
        return f"EARNINGS_IN_HOLDING_WINDOW: {earnings.date()} (gap risk; stop/target may be skipped)"
    return f"next earnings {earnings.date()} (outside {max_holding_days}-session window)"


def earnings_status(earnings: pd.Timestamp | None, signal_date: pd.Timestamp, max_holding_days: int) -> str:
    """UNKNOWN (no date, or only a past date) | IN_HOLDING_WINDOW | OUTSIDE_HOLDING_WINDOW. Unknown is never treated as clear."""
    if earnings is None or earnings < signal_date:
        return "UNKNOWN"
    horizon = signal_date + pd.Timedelta(days=int(max_holding_days * 1.5) + 1)
    return "IN_HOLDING_WINDOW" if earnings <= horizon else "OUTSIDE_HOLDING_WINDOW"


def news_link(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker}/news"


def _parse_item(item: dict) -> dict | None:
    """Handle both yfinance news layouts (nested 'content' and flat legacy). Returns None if no title."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = c.get("title")
    if not title:
        return None
    publisher = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
    url = ((c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url")
           if isinstance(c.get("canonicalUrl") or c.get("clickThroughUrl"), dict) else c.get("link"))
    when = c.get("pubDate") or c.get("providerPublishTime")
    if isinstance(when, (int, float)):
        when = pd.Timestamp(when, unit="s", tz="UTC").isoformat()
    return {"title": str(title), "publisher": publisher or "", "published": str(when or ""), "url": url or ""}


def recent_headlines(ticker: str, n: int = 3, fetch=None) -> dict:
    """Latest Yahoo headlines, DISPLAY ONLY (never scored). An empty/failed feed is NOT_AVAILABLE - that is not evidence
    that there is no news, so the link is always provided for a manual check."""
    out = {"status": "NOT_AVAILABLE", "headlines": [], "link": news_link(ticker),
           "note": "Yahoo returned no headlines; this does NOT mean there is no news - check the link"}
    try:
        if fetch is None:
            import yfinance as yf
            raw = yf.Ticker(ticker).news
        else:
            raw = fetch(ticker)
        items = [h for h in (_parse_item(i) for i in (raw or []) if isinstance(i, dict)) if h][:n]
    except Exception as exc:
        out["note"] = f"headline fetch failed ({type(exc).__name__}); check the link"
        return out
    if items:
        out.update(status="OK", headlines=items, note="Headlines are display only and are not part of the score")
    return out
