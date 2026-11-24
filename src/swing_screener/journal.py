"""Paper-trading journal: a forward test that nobody can have tuned to.

`scan --journal` records EVERY qualified candidate of the scan's market-bar date (not only the top N shown), with the
rules in force at that moment. `journal` then follows each one with the backtest's own entry and exit rules (next
session's open, ATR stop/target re-based on that open, gaps fill at the open, stop checked before target, time exit,
optional entry filter and breakeven stop) and reports per-trade results in R (multiples of the initial risk per share,
costs included).

Each candidate is followed on its own: there is no $1,000 account, no 5-position limit and no share rounding, so the
report measures signal quality, not portfolio returns. Nothing here places or suggests orders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import bar_exit, breakeven_price, buy_fill, sell_fill

WIN_EPS_R = 1e-9          # a breakeven exit nets ~0R; float noise above zero must not count as a win
MIN_TRADES_TO_JUDGE = 100

COLUMNS = ["recorded_at_utc", "signal_date", "ticker", "rank", "in_top_n", "setups", "score", "plan_status",
           "unverified_because", "earnings_status", "signal_close", "atr14", "top_n", "atr_stop_multiple",
           "atr_target_multiple", "max_holding_days", "max_entry_gap_atr", "breakeven_trigger_atr", "cost_rate",
           "min_score", "enabled_setups", "regime_sma", "rules_hash"]


def rules_hash(s) -> str:
    """Short fingerprint of every setting, so a report can show when the rules changed mid-journal."""
    blob = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(s.__dict__.items())}, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def journal_rows(picks: pd.DataFrame, s, recorded_at: str, top_n: int | None = None) -> pd.DataFrame:
    """Journal rows from a full scan output (every qualified candidate, ranked). Exit rules are stored per row so later
    config edits never change how an already-recorded pick is scored."""
    if picks.empty:
        return pd.DataFrame(columns=COLUMNS)
    top_n = s.top_n if top_n is None else top_n
    h = rules_hash(s)
    rows = [{"recorded_at_utc": recorded_at, "signal_date": p["signal_date"], "ticker": p["ticker"], "rank": int(p["rank"]),
             "in_top_n": int(p["rank"]) <= top_n, "setups": ";".join(p["setups"]), "score": p["score"],
             "plan_status": p["plan_status"], "unverified_because": p["unverified_because"],
             "earnings_status": p["earnings_status"], "signal_close": p["close"], "atr14": p["atr14"], "top_n": top_n,
             "atr_stop_multiple": s.atr_stop_multiple, "atr_target_multiple": s.atr_target_multiple,
             "max_holding_days": s.max_holding_days, "max_entry_gap_atr": s.max_entry_gap_atr,
             "breakeven_trigger_atr": s.breakeven_trigger_atr, "cost_rate": s.cost_rate, "min_score": s.min_score,
             "enabled_setups": ";".join(s.enabled_setups), "regime_sma": s.regime_sma, "rules_hash": h}
            for _, p in picks.iterrows()]
    return pd.DataFrame(rows, columns=COLUMNS)


def load_journal(directory: str | Path) -> pd.DataFrame:
    p = Path(directory) / "picks.csv"
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype={"signal_date": str, "ticker": str, "setups": str, "unverified_because": str},
                       keep_default_na=False, na_values=[""])


def append_to_journal(directory: str | Path, rows: pd.DataFrame, signal_date: str) -> tuple[int, str]:
    """Append one scan date's candidates. A date already in the journal is never overwritten (the first recording is
    what you would have acted on). Returns (rows added, message)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    existing = load_journal(d)
    if (existing["signal_date"].astype(str) == signal_date).any():
        return 0, f"journal already has signal date {signal_date}; nothing added (the first recording is kept)"
    out = pd.concat([existing, rows], ignore_index=True) if len(existing) else rows
    out.to_csv(d / "picks.csv", index=False)
    if rows.empty:
        return 0, f"no qualified candidates on {signal_date}; nothing to record"
    return len(rows), f"recorded {len(rows)} candidate(s) for {signal_date} in {d / 'picks.csv'}"


def follow_pick(row, frame: pd.DataFrame | None, calendar: pd.DatetimeIndex) -> dict:
    """Outcome of one recorded candidate, using only completed bars. Sessions are counted on the benchmark calendar
    (a missing ticker bar still counts, as in the backtest). Prices are raw; R and returns include costs."""
    base = {"status": "awaiting_entry", "entry_date": "", "entry_price": np.nan, "stop": np.nan, "target": np.nan,
            "exit_date": "", "exit_price": np.nan, "exit_reason": "", "sessions_held": 0, "breakeven_armed": False,
            "r_multiple": np.nan, "return_pct": np.nan}
    signal = pd.Timestamp(row["signal_date"])
    after = calendar[calendar > signal]
    if len(after) == 0:
        return base
    entry_day = after[0]
    if frame is None or entry_day not in frame.index:
        return {**base, "status": "no_entry_bar", "entry_date": entry_day.date().isoformat()}
    o = float(frame.at[entry_day, "Open"])
    atr, rate = float(row["atr14"]), float(row["cost_rate"])
    if row["max_entry_gap_atr"] > 0 and o > row["signal_close"] + row["max_entry_gap_atr"] * atr:
        return {**base, "status": "skipped_gap_above_max_entry", "entry_date": entry_day.date().isoformat(), "entry_price": o}
    stop0, target = o - row["atr_stop_multiple"] * atr, o + row["atr_target_multiple"] * atr
    if not (atr > 0 and stop0 > 0):
        return {**base, "status": "invalid_levels", "entry_date": entry_day.date().isoformat(), "entry_price": o}
    out = {**base, "status": "open", "entry_date": entry_day.date().isoformat(), "entry_price": o, "stop": stop0,
           "target": target}
    stop, armed, last_close, held = stop0, False, o, 0
    for n, d in enumerate(after, start=1):
        held = n
        if d not in frame.index:
            continue
        bo, bh, bl, bc = (float(frame.at[d, k]) for k in ("Open", "High", "Low", "Close"))
        last_close = bc
        hit = bar_exit(bo, bh, bl, stop, target)
        if hit:
            reason, price, _ = hit
            reason = f"breakeven_{reason}" if armed and reason.startswith("stop") else reason
            out.update(status="closed", exit_date=d.date().isoformat(), exit_price=price, exit_reason=reason)
            break
        if n >= row["max_holding_days"]:
            out.update(status="closed", exit_date=d.date().isoformat(), exit_price=bc, exit_reason="time")
            break
        if row["breakeven_trigger_atr"] > 0 and not armed and bh >= o + row["breakeven_trigger_atr"] * atr:
            stop, armed = max(stop, breakeven_price(o, rate)), True
    exit_px = out["exit_price"] if out["status"] == "closed" else last_close   # open trades: marked at the last close
    ef, xf = buy_fill(o, rate), sell_fill(exit_px, rate)
    out.update(sessions_held=held, breakeven_armed=armed,
               r_multiple=(xf - ef) / (ef - sell_fill(stop0, rate)), return_pct=(xf / ef - 1) * 100)
    return out


def compute_outcomes(journal: pd.DataFrame, frames: dict, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if journal.empty:
        return journal.assign(status=[])
    res = [follow_pick(r, frames.get(r["ticker"]), calendar) for _, r in journal.iterrows()]
    return pd.concat([journal.reset_index(drop=True), pd.DataFrame(res)], axis=1)


def _top_n_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Re-rank within each signal date (by the recorded rank) and keep that day's top_n."""
    if df.empty:
        return df
    df = df.sort_values(["signal_date", "rank"])
    return df[df.groupby("signal_date").cumcount() < df["top_n"]]


GROUPS = {
    "scan_top_n": ("What the scan showed: each day's top N", lambda o: o[o["in_top_n"].astype(bool)]),
    "verified_top_n": ("Only VERIFIED_PLAN picks, top N per day",
                       lambda o: _top_n_by_day(o[o["plan_status"] == "VERIFIED_PLAN"])),
    "no_relative_strength_top_n": ("Variant V3: picks with a breakout or pullback setup, top N per day",
                                   lambda o: _top_n_by_day(o[o["setups"].str.contains("breakout|pullback", regex=True)])),
    "all_qualified": ("Every qualified candidate", lambda o: o),
}


def summarize(outcomes: pd.DataFrame) -> list[dict]:
    """Per-group statistics on CLOSED trades (open trades are counted but not scored)."""
    rows = []
    for key, (label, pick) in GROUPS.items():
        g = pick(outcomes) if len(outcomes) else outcomes
        closed = g[g["status"] == "closed"] if len(g) else g
        r = closed["r_multiple"].astype(float) if len(closed) else pd.Series(dtype=float)
        wins, losses = r[r > WIN_EPS_R], r[r <= WIN_EPS_R]
        avg_win, avg_loss = (float(wins.mean()) if len(wins) else None), (float(losses.mean()) if len(losses) else None)
        payoff = avg_win / -avg_loss if avg_win is not None and avg_loss is not None and avg_loss < 0 else None
        gl = float(-losses.sum())
        rows.append({
            "group": key, "description": label,
            "recorded": len(g), "closed": len(closed),
            "open": int((g["status"] == "open").sum()) if len(g) else 0,
            "awaiting_entry": int((g["status"] == "awaiting_entry").sum()) if len(g) else 0,
            "not_entered": int(g["status"].isin(["no_entry_bar", "skipped_gap_above_max_entry", "invalid_levels"]).sum()) if len(g) else 0,
            "win_rate_pct": float(len(wins) / len(r) * 100) if len(r) else None,
            "avg_win_r": avg_win, "avg_loss_r": avg_loss, "payoff_ratio": payoff,
            "breakeven_win_rate_pct": 100 / (1 + payoff) if payoff else None,
            "profit_factor": float(wins.sum()) / gl if gl > 0 else None,
            "expectancy_r": float(r.mean()) if len(r) else None,
            "expectancy_pct": float(closed["return_pct"].mean()) if len(closed) else None,
            "enough_trades": len(r) >= MIN_TRADES_TO_JUDGE,
        })
    return rows
