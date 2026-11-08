"""Event-driven daily-bar simulator + sequential walk-forward evaluation.

Timeline per session d (all decisions use information available at that point):
  1. Pending entries (signals generated at the CLOSE of the previous session) are filled at d's OPEN, sized from the
     equity marked at the previous close and the cash actually available.
  2. Every open position (including one entered this session) is checked against d's bar:
       open <= stop -> exit at the open (gap through stop);   open >= target -> exit at the open (gap through target);
       low <= stop  -> exit at the stop (checked BEFORE the target, so a bar touching both counts as a loss);
       high >= target -> exit at the target;  sessions held >= max_holding_days -> exit at the close.
     Optional breakeven stop (breakeven_trigger_atr = k > 0): if the bar's high reached entry + k*ATR and the position is
     still open, the stop is raised to the cost-inclusive breakeven price from the NEXT session (daily bars cannot show
     whether the high came before the low, so it never applies to the triggering bar). The stop is never lowered.
  3. Positions are marked at the latest known close (no bar => last close is carried and counted as a missing-bar day).
  4. Signals are generated from d's close and scheduled for the next session. Not generated on the final session.
  5. On the final session all open positions are liquidated at the close WITH costs and the final equity row reflects it.

Indicators are rolling/causal (see indicators.py; truncation-invariance is unit-tested), so precomputing them over the full
history cannot leak future data into a decision.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .risk import buy_fill, levels, sell_fill, size_position
from .screener import rank_candidates


WIN_EPS_USD = 1e-6   # a breakeven exit nets ~0; float noise above zero must not count as a win


class InsufficientHistory(ValueError):
    pass


@dataclass
class SimResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    skipped: pd.DataFrame


def precompute_signals(indicators: dict, s, dates) -> dict:
    """Ranked, de-duplicated candidates for each signal date (independent of costs and portfolio state)."""
    return {d: rank_candidates(indicators, s, as_of=d, dedupe=True) for d in dates}


def simulate(prices: dict, signals: dict, dates: pd.DatetimeIndex, s) -> SimResult:
    rate = s.cost_rate
    comm_rate = s.commission_bps_per_side / 10_000.0
    bars = {t: {d: (float(r.Open), float(r.High), float(r.Low), float(r.Close))
                for d, r in zip(f.index, f[["Open", "High", "Low", "Close"]].itertuples(index=False))}
            for t, f in prices.items()}
    cash = float(s.initial_capital)
    prev_equity = cash
    positions: dict = {}
    pending: dict = {}
    trades, eq_rows, skipped = [], [], []
    last_close: dict = {}
    n = len(dates)

    def close_position(t, pos, date, raw_price, reason, extra=None):
        nonlocal cash
        fill = sell_fill(raw_price, rate)
        proceeds = fill * pos["shares"]
        cash += proceeds
        notional = raw_price * pos["shares"]
        pnl = proceeds - pos["entry_outlay"]
        trades.append({
            "ticker": t, "setup": pos["setup"], "score": pos["score"], "signal_date": pos["signal_date"],
            "entry_date": pos["entry_date"].date().isoformat(), "entry_price": pos["entry_price"],
            "entry_fill": pos["entry_fill"], "shares": pos["shares"], "stop": pos["initial_stop"], "exit_stop": pos["stop"],
            "target": pos["target"],
            "exit_date": date.date().isoformat(), "exit_price": raw_price, "exit_fill": fill, "exit_reason": reason,
            "sessions_held": pos["sessions"], "pnl": pnl, "return_pct": pnl / pos["entry_outlay"] * 100,
            "entry_notional": pos["entry_notional"], "exit_notional": notional,
            "commission_usd": pos["entry_notional"] * comm_rate + notional * comm_rate,
            "slippage_usd": (pos["entry_notional"] + notional) * (rate - comm_rate),
            "ambiguous_bar": bool(extra and extra.get("ambiguous")), "missing_bar_days": pos["missing_bar_days"],
            "breakeven_armed": pos["breakeven_armed"],
        })

    for i, d in enumerate(dates):
        # 1. entries at the open
        for c in pending.pop(d, []):
            t = c["ticker"]
            if t in positions:
                skipped.append({"date": d, "ticker": t, "reason": "duplicate_position"}); continue
            if len(positions) >= s.max_positions:
                skipped.append({"date": d, "ticker": t, "reason": "max_positions"}); continue
            bar = bars.get(t, {}).get(d)
            if bar is None:
                skipped.append({"date": d, "ticker": t, "reason": "no_bar_on_entry_session"}); continue
            o = bar[0]
            if s.max_entry_gap_atr > 0:
                if "close" not in c:
                    raise ValueError("entry filter needs the signal-day 'close' on each candidate")
                if o > c["close"] + s.max_entry_gap_atr * c["_atr"]:     # open is too far above the signal close
                    skipped.append({"date": d, "ticker": t, "reason": "gap_above_max_entry"}); continue
            lv = levels(o, c["_atr"], s)
            if not lv["valid"]:
                skipped.append({"date": d, "ticker": t, "reason": "invalid_levels"}); continue
            sz = size_position(prev_equity, cash, o, lv["stop"], s)
            if sz["shares"] < 1:
                skipped.append({"date": d, "ticker": t, "reason": f"zero_shares({sz['reason']})"}); continue
            outlay = sz["shares"] * sz["buy_fill"]
            cash -= outlay
            positions[t] = {"setup": c["setup"], "score": c["score"], "signal_date": c["signal_date"],
                            "entry_idx": i, "entry_date": d, "entry_price": o, "entry_fill": sz["buy_fill"],
                            "shares": sz["shares"], "stop": lv["stop"], "initial_stop": lv["stop"], "target": lv["target"],
                            "entry_outlay": outlay, "entry_notional": o * sz["shares"],
                            "sessions": 1, "missing_bar_days": 0, "atr": c["_atr"], "breakeven_armed": False}
            last_close[t] = o
        # 2. exits
        for t in list(positions):
            pos = positions[t]
            pos["sessions"] = i - pos["entry_idx"] + 1
            bar = bars.get(t, {}).get(d)
            if bar is None:
                pos["missing_bar_days"] += 1
                continue
            o, h, l, c = bar
            last_close[t] = c
            be = "breakeven_" if pos["breakeven_armed"] else ""
            if o <= pos["stop"]:
                close_position(t, pos, d, o, f"{be}stop_gap")
            elif o >= pos["target"]:
                close_position(t, pos, d, o, "target_gap")
            elif l <= pos["stop"]:
                close_position(t, pos, d, pos["stop"], f"{be}stop", {"ambiguous": h >= pos["target"]})
            elif h >= pos["target"]:
                close_position(t, pos, d, pos["target"], "target")
            elif pos["sessions"] >= s.max_holding_days:
                close_position(t, pos, d, c, "time")
            else:
                # Arm after this bar's exit checks, so the raised stop first applies to the next session.
                if (s.breakeven_trigger_atr > 0 and not pos["breakeven_armed"]
                        and h >= pos["entry_price"] + s.breakeven_trigger_atr * pos["atr"]):
                    pos["stop"] = max(pos["stop"], pos["entry_price"] * (1 + rate) / (1 - rate))
                    pos["breakeven_armed"] = True
                continue
            del positions[t]
        # 3. mark to market
        mv = sum(p["shares"] * last_close[t] for t, p in positions.items())
        pre_liq = cash + mv
        # 5. final-session liquidation (before recording the closing equity)
        if i == n - 1:
            for t in list(positions):
                pos = positions[t]
                has_bar = d in bars.get(t, {})
                close_position(t, pos, d, last_close[t], "end_of_test" if has_bar else "end_of_test_no_bar")
                del positions[t]
            mv = 0.0
        eq = cash + mv
        eq_rows.append({"date": d, "cash": cash, "market_value": mv, "equity": eq, "equity_pre_liquidation": pre_liq,
                        "open_positions": len(positions)})
        prev_equity = eq
        # 4. signals from this close, for the next session
        if i < n - 1:
            cands = signals.get(d)
            if cands is not None and len(cands):
                pending[dates[i + 1]] = [r for r in cands.to_dict("records") if r["ticker"] not in positions]
    res = SimResult(pd.DataFrame(trades), pd.DataFrame(eq_rows),
                    pd.DataFrame(skipped, columns=["date", "ticker", "reason"]))
    assert_reconciled(res, s.initial_capital)
    return res


def assert_reconciled(res: SimResult, start_equity: float, tol: float = 1e-6) -> None:
    """Accounting invariant: after final liquidation, ending equity == start equity + sum of trade P&L, no open positions."""
    final = float(res.equity["equity"].iloc[-1])
    pnl = float(res.trades["pnl"].sum()) if len(res.trades) else 0.0
    if res.equity["open_positions"].iloc[-1] != 0 or abs(final - (start_equity + pnl)) > tol * max(1.0, start_equity):
        raise AssertionError(f"ledger does not reconcile: final equity {final} vs start+pnl {start_equity + pnl}")


# ------------------------------------------------------------------ metrics
def _max_dd(values) -> float:
    v = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1).min() * 100)


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, start_equity: float) -> dict:
    eq = equity["equity"].to_numpy(float)
    end_eq = float(eq[-1])
    days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days + 1
    tr = end_eq / start_equity - 1
    m = {"start_date": equity["date"].iloc[0].date().isoformat(), "end_date": equity["date"].iloc[-1].date().isoformat(),
         "calendar_days": days, "start_equity": start_equity, "end_equity": end_eq,
         "total_return_pct": tr * 100,
         "annualized_return_pct": ((1 + tr) ** (365.25 / days) - 1) * 100 if days > 0 and 1 + tr > 0 else None,
         "annualization_note": "extrapolated from a period shorter than one year; not a forecast" if days < 365 else "",
         "max_drawdown_pct": _max_dd(np.concatenate([[start_equity], eq])),
         "exposure_pct": float((equity["market_value"] / equity["equity"]).mean() * 100),
         "trades": len(trades)}
    avg_eq = float(np.mean(eq))
    if len(trades):
        w, lo = trades[trades.pnl > WIN_EPS_USD], trades[trades.pnl <= WIN_EPS_USD]
        gp, gl = float(w.pnl.sum()), float(-lo.pnl.sum())
        m.update({"win_rate_pct": len(w) / len(trades) * 100,
                  "avg_win_usd": float(w.pnl.mean()) if len(w) else 0.0,
                  "avg_loss_usd": float(lo.pnl.mean()) if len(lo) else 0.0,
                  "profit_factor": gp / gl if gl > 0 else None,
                  "expectancy_usd_after_costs": float(trades.pnl.mean()),
                  "expectancy_pct_after_costs": float(trades.return_pct.mean()),
                  "total_commission_usd": float(trades.commission_usd.sum()),
                  "total_slippage_usd": float(trades.slippage_usd.sum()),
                  "turnover_x_avg_equity": float((trades.entry_notional.sum() + trades.exit_notional.sum()) / avg_eq),
                  "avg_sessions_held": float(trades.sessions_held.mean())})
    else:
        m.update({"win_rate_pct": None, "avg_win_usd": None, "avg_loss_usd": None, "profit_factor": None,
                  "expectancy_usd_after_costs": None, "expectancy_pct_after_costs": None,
                  "total_commission_usd": 0.0, "total_slippage_usd": 0.0, "turnover_x_avg_equity": 0.0,
                  "avg_sessions_held": None})
    return m


def spy_buy_and_hold(benchmark: pd.DataFrame, dates: pd.DatetimeIndex, s) -> dict:
    """Fractional-share SPY buy at the first OOS open and sell at the last OOS close, paying the same per-side costs."""
    b = benchmark.loc[dates[0]:dates[-1]]
    rate = s.cost_rate
    units = s.initial_capital / buy_fill(float(b.Open.iloc[0]), rate)
    curve = units * b.Close.to_numpy(float)
    final = units * sell_fill(float(b.Close.iloc[-1]), rate)
    curve[-1] = final
    days = (b.index[-1] - b.index[0]).days + 1
    tr = final / s.initial_capital - 1
    return {"total_return_pct": tr * 100, "end_equity": float(final),
            "annualized_return_pct": ((1 + tr) ** (365.25 / days) - 1) * 100 if 1 + tr > 0 else None,
            "max_drawdown_pct": _max_dd(np.concatenate([[s.initial_capital], curve])),
            "basis": "buy first OOS open, sell last OOS close, same per-side costs as strategy"}


def make_windows(calendar: pd.DatetimeIndex, s) -> list:
    """Sequential non-overlapping OOS windows after `warmup_years` of history. Returns [(start, end, partial)]."""
    first = calendar[0] + pd.DateOffset(years=s.warmup_years)
    oos = calendar[calendar >= first]
    if len(oos) == 0:
        raise InsufficientHistory(f"history ({calendar[0].date()}..{calendar[-1].date()}) shorter than warm-up {s.warmup_years}y")
    wins, k = [], 0
    while True:
        b0 = first + pd.DateOffset(months=s.test_months * k)
        b1 = first + pd.DateOffset(months=s.test_months * (k + 1))
        if b0 > calendar[-1]:
            break
        idx = calendar[(calendar >= b0) & (calendar < b1)]
        if len(idx):
            wins.append((idx[0], idx[-1], b1 > calendar[-1] + pd.Timedelta(days=1)))
        k += 1
    if not wins or all(w[2] for w in wins):
        raise InsufficientHistory("no complete out-of-sample window available after warm-up")
    return wins


def window_table(res: SimResult, benchmark: pd.DataFrame, wins, s) -> pd.DataFrame:
    rows, prev_eq, prev_spy_close = [], float(s.initial_capital), None
    eq, tr = res.equity, res.trades
    for (a, b, partial) in wins:
        e = eq[(eq.date >= a) & (eq.date <= b)]
        t = tr[(pd.to_datetime(tr.exit_date) >= a) & (pd.to_datetime(tr.exit_date) <= b)] if len(tr) else tr
        m = compute_metrics(t, e, prev_eq) if len(e) else {}
        bw = benchmark.loc[a:b]
        base = prev_spy_close if prev_spy_close is not None else float(bw.Open.iloc[0])
        spy_ret = (float(bw.Close.iloc[-1]) / base - 1) * 100
        rows.append({"window_start": a.date().isoformat(), "window_end": b.date().isoformat(), "partial": partial,
                     **{k: m.get(k) for k in ("total_return_pct", "max_drawdown_pct", "trades", "win_rate_pct",
                                              "profit_factor", "expectancy_usd_after_costs", "exposure_pct",
                                              "total_commission_usd", "total_slippage_usd", "turnover_x_avg_equity")},
                     "start_equity": prev_eq, "end_equity": m.get("end_equity"),
                     "spy_return_pct_price_only": spy_ret})
        prev_eq = float(e.equity.iloc[-1]); prev_spy_close = float(bw.Close.iloc[-1])
    return pd.DataFrame(rows)


def run_walk_forward(prices: dict, benchmark: pd.DataFrame, s, *, output_dir: str | Path | None = None,
                     universe_info: dict | None = None, data_reports=None) -> dict:
    """Fixed-rule sequential walk-forward. No parameter is fit on any data: warm-up exists only so indicators and the
    200-session average are defined before the first out-of-sample session."""
    if not prices:
        raise InsufficientHistory("no valid price histories supplied")
    cal = benchmark.index
    wins = make_windows(cal, s)
    oos_dates = cal[(cal >= wins[0][0]) & (cal <= wins[-1][1])]
    ind = {t: add_indicators(f, s, benchmark) for t, f in prices.items()}
    signals = precompute_signals(ind, s, oos_dates)
    base = simulate(prices, signals, oos_dates, s)
    combined = compute_metrics(base.trades, base.equity, s.initial_capital)
    spy = spy_buy_and_hold(benchmark, oos_dates, s)
    win_df = window_table(base, benchmark, wins, s)
    sens = []
    for mult in s.cost_sensitivity_multipliers:
        r = base if mult == 1.0 else simulate(prices, signals, oos_dates, s.with_cost_multiplier(mult))
        m = compute_metrics(r.trades, r.equity, s.initial_capital)
        sens.append({"cost_multiplier": mult, "per_side_cost_bps": (s.commission_bps_per_side + s.slippage_bps_per_side) * mult,
                     **{k: m.get(k) for k in ("total_return_pct", "max_drawdown_pct", "trades", "profit_factor",
                                              "expectancy_usd_after_costs", "end_equity")}})
    entry_tests = []
    for k in (0.0, *s.entry_filter_tests):
        r = base if k == s.max_entry_gap_atr else simulate(prices, signals, oos_dates, replace(s, max_entry_gap_atr=k))
        m = compute_metrics(r.trades, r.equity, s.initial_capital)
        skipped_gap = int((r.skipped.reason == "gap_above_max_entry").sum()) if len(r.skipped) else 0
        entry_tests.append({"max_entry_gap_atr": k, "label": "baseline: buy every next open" if k == 0 else
                            ("PRIMARY (fixed in advance)" if k == 0.5 else "context only"),
                            "skipped_for_gap": skipped_gap,
                            **{key: m.get(key) for key in ("total_return_pct", "max_drawdown_pct", "trades", "win_rate_pct",
                                                           "profit_factor", "expectancy_usd_after_costs", "end_equity")}})
    summary = {
        "status": "RESEARCH_ONLY_NOT_VALIDATED",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "oos_period": {"start": oos_dates[0].date().isoformat(), "end": oos_dates[-1].date().isoformat(),
                       "warmup_years": s.warmup_years, "test_months": s.test_months, "n_windows": len(wins)},
        "universe_size_used": len(prices),
        "strategy_combined_oos": combined, "spy_buy_and_hold_oos": spy,
        "excess_return_vs_spy_pct_points": combined["total_return_pct"] - spy["total_return_pct"],
        "by_setup": ({k: {"trades": len(g), "pnl_usd": float(g.pnl.sum()), "win_rate_pct": float((g.pnl > WIN_EPS_USD).mean() * 100),
                          "avg_return_pct": float(g.return_pct.mean())} for k, g in base.trades.groupby("setup")}
                     if len(base.trades) else {}),
        "skipped_entries": base.skipped.reason.str.replace(r"\(.*\)", "", regex=True).value_counts().to_dict()
        if len(base.skipped) else {},
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.__dict__.items()},
        "universe": universe_info or {},
        "caveats": [
            "Survivorship bias: tickers are today's listings; delisted/bankrupt names are absent, inflating results.",
            "Universe selection used today's liquidity; this is look-ahead in membership.",
            "Prices are split-adjusted, not dividend-adjusted (executable-style); dividends are ignored for both strategy and SPY.",
            "Daily bars cannot show intraday ordering; stop-before-target is a pessimistic convention. Fills at stop/target "
            "assume no further slippage beyond the configured bps.",
            "No earnings/event filter. Costs are configured estimates, not broker quotes.",
            "A positive result on a short OOS period with few trades is not evidence of an edge.",
        ],
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        base.trades.to_csv(out / "trades.csv", index=False)
        base.equity.to_csv(out / "equity.csv", index=False)
        base.skipped.to_csv(out / "skipped_entries.csv", index=False)
        win_df.to_csv(out / "windows.csv", index=False)
        pd.DataFrame(sens).to_csv(out / "cost_sensitivity.csv", index=False)
        summary["cost_sensitivity"] = sens
        summary["entry_filter_comparison"] = entry_tests
        pd.DataFrame(entry_tests).to_csv(out / "entry_filter.csv", index=False)
        summary["windows"] = json.loads(win_df.to_json(orient="records"))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    else:
        summary["entry_filter_comparison"] = entry_tests
        summary["cost_sensitivity"] = sens
        summary["windows"] = json.loads(win_df.to_json(orient="records"))
    return summary


def universe_fingerprint(tickers) -> str:
    return hashlib.sha256("\n".join(sorted(tickers)).encode()).hexdigest()[:16]
