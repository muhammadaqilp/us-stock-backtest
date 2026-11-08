"""Pre-registered rule-variant comparison (see README "Rule variants"). Research only.

Runs the unchanged walk-forward for the baseline and each variant over two periods and writes one table:
  holdout: 2011-01-01..2021-11-02 from data/cache_holdout_2011 (never used to design or tune any rule)
  recent:  the normal data/cache history (2023-26 OOS; V2/V3 were derived from it, so it is context only)
  holdout2001: 2001-01-01..2012-12-31 from data/cache_holdout_2001 (breakeven-stop test, README "Breakeven stop")

    python research/run_variants.py [--set round1|breakeven] [--period holdout|recent|holdout2001|both|all] [--max-tickers 400]
"""
import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swing_screener.backtest import run_walk_forward
from swing_screener.config import load_settings
from swing_screener.data import YFinanceProvider, get_history, load_universe_history
from swing_screener.universe import load_universe, sample_universe

VARIANTS = {
    "baseline": {},
    "V1_regime": {"regime_sma": 200},
    "V2_score75": {"min_score": 75.0},
    "V3_no_rs": {"enabled_setups": ("breakout", "pullback")},
    "V4_primary": {"regime_sma": 200, "min_score": 75.0, "enabled_setups": ("breakout", "pullback")},
    "T1_target2atr_illustration": {"atr_target_multiple": 2.0, "min_reward_risk": 1.0},
}
BREAKEVEN = {
    "baseline": {},
    "B1_breakeven_1atr": {"breakeven_trigger_atr": 1.0},
    "V3_no_rs": {"enabled_setups": ("breakout", "pullback")},
    "B1_plus_V3_context": {"breakeven_trigger_atr": 1.0, "enabled_setups": ("breakout", "pullback")},
}
SETS = {"round1": VARIANTS, "breakeven": BREAKEVEN}
PERIODS = {
    "holdout": {"cache_dir": "data/cache_holdout_2011", "start": "2011-01-01", "end": "2021-11-02"},
    "recent": {"cache_dir": "data/cache", "start": None, "end": None},
    "holdout2001": {"cache_dir": "data/cache_holdout_2001", "start": "2001-01-01", "end": "2013-01-01"},
}


def load_period(base, name, n):
    p = PERIODS[name]
    s = replace(base, cache_dir=p["cache_dir"])
    start = (pd.Timestamp(p["start"]) if p["start"] else
             (pd.Timestamp.now().normalize() - pd.DateOffset(years=s.history_years)))
    end = pd.Timestamp(p["end"]) if p["end"] else None
    prov = YFinanceProvider()
    spy = get_history("SPY", prov, s, start=start, end=end, check_stale=False)
    tickers = sample_universe(load_universe(), n, s.sample_seed)
    frames, reports = load_universe_history(tickers, prov, s, start=start, end=end, calendar=spy.frame.index,
                                            check_stale=False)
    return spy.frame, frames, sum(not r.ok for r in reports)


def bootstrap_ci(x, n=5000, seed=0):
    """90% bootstrap interval of the mean per-trade return (%)."""
    if len(x) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = rng.choice(np.asarray(x, float), size=(n, len(x))).mean(axis=1)
    return tuple(round(float(v), 3) for v in np.percentile(means, [5, 95]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="round1")
    ap.add_argument("--period", choices=[*PERIODS, "both", "all"], default="both")
    ap.add_argument("--max-tickers", type=int, default=400)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    base = replace(load_settings("config.toml"), entry_filter_tests=(), cost_sensitivity_multipliers=(1.0, 2.0))
    out = Path(args.output_dir or f"reports/variants_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    periods = {"both": ["holdout", "recent"], "all": ["holdout2001", "holdout", "recent"]}.get(args.period, [args.period])
    for period in periods:
        spy, frames, failed = load_period(base, period, args.max_tickers)
        print(f"[{period}] {len(frames)} tickers usable, {failed} failed validation; SPY {spy.index[0].date()}..{spy.index[-1].date()}",
              flush=True)
        for name, kw in SETS[args.set].items():
            s = replace(base, **kw)
            sm = run_walk_forward(frames, spy, s, output_dir=out / period / name)
            c = sm["strategy_combined_oos"]
            x2 = next(r for r in sm["cost_sensitivity"] if r["cost_multiplier"] == 2.0)
            tr = pd.read_csv(out / period / name / "trades.csv") if c["trades"] else pd.DataFrame({"return_pct": []})
            lo, hi = bootstrap_ci(tr["return_pct"])
            row = {"period": period, "variant": name, "oos_start": sm["oos_period"]["start"], "oos_end": sm["oos_period"]["end"],
                   "trades": c["trades"], "win_rate_pct": c["win_rate_pct"],
                   "expectancy_pct": c["expectancy_pct_after_costs"], "exp_pct_ci90_lo": lo, "exp_pct_ci90_hi": hi,
                   "expectancy_usd": c["expectancy_usd_after_costs"], "profit_factor": c["profit_factor"],
                   "profit_factor_2x_costs": x2["profit_factor"], "total_return_pct": c["total_return_pct"],
                   "max_drawdown_pct": c["max_drawdown_pct"], "exposure_pct": c["exposure_pct"],
                   "spy_return_pct": sm["spy_buy_and_hold_oos"]["total_return_pct"],
                   "breakeven_exits": int(tr["exit_reason"].astype(str).str.startswith("breakeven").sum()) if c["trades"] else 0,
                   "by_setup": json.dumps({k: [v["trades"], round(v["avg_return_pct"], 3)] for k, v in sm["by_setup"].items()})}
            rows.append(row)
            print(f"  {name:28s} trades {row['trades']:4d}  win {row['win_rate_pct'] or 0:5.1f}%  exp {row['expectancy_pct'] or 0:+.3f}% "
                  f"[{lo}, {hi}]  PF {row['profit_factor'] or 0:.3f}  PF2x {row['profit_factor_2x_costs'] or 0:.3f}  "
                  f"ret {row['total_return_pct']:+.2f}%  DD {row['max_drawdown_pct']:.2f}%", flush=True)
            pd.DataFrame(rows).to_csv(out / "variants.csv", index=False)
    print(f"wrote {out / 'variants.csv'}")


if __name__ == "__main__":
    main()
