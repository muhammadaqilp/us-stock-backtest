"""Manual CLI. No scheduling, notifications, broker access or order placement exist in this project."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .backtest import InsufficientHistory, run_walk_forward
from .config import load_settings
from .data import (
    DataProviderError,
    YFinanceProvider,
    get_history,
    load_universe_history,
    reports_to_frame,
)
from .events import next_earnings_date, recent_headlines
from .picks import DISCLAIMER, StaleMarketData, build_daily_picks
from .plan import build_plan, plan_at_entry, plan_steps
from .screener import PRICE_LABEL
from .universe import (
    build_universe,
    load_names,
    load_universe,
    sample_universe,
    snapshot,
)


def _history_start(s) -> pd.Timestamp:
    return (pd.Timestamp.now().normalize() - pd.DateOffset(years=s.history_years)).tz_localize(None)


def _tickers(args, s) -> list:
    if args.tickers:
        return sorted({t.strip().upper() for t in args.tickers.split(",") if t.strip()} - {"SPY"})
    n = args.max_tickers if args.max_tickers is not None else s.max_tickers
    return sample_universe(load_universe(args.universe_file), n, s.sample_seed)


def _load(args, s, provider, *, scan: bool):
    start = _history_start(s)
    spy = get_history("SPY", provider, s, start=start, refresh=args.refresh,
                      max_cache_age_hours=s.scan_cache_max_age_hours if scan else None)
    if not spy.report.ok:
        raise DataProviderError(f"SPY benchmark failed validation: {'; '.join(spy.report.errors)}")
    if scan and spy.report.stale:
        raise DataProviderError("SPY data is stale; refusing to produce a scan (use --refresh)")
    tickers = _tickers(args, s)
    frames, reports = load_universe_history(
        tickers, provider, s, start=start, refresh=args.refresh, calendar=spy.frame.index,
        max_cache_age_hours=s.scan_cache_max_age_hours if scan else None, check_stale=scan)
    return spy, tickers, frames, reports


def _write_quality(reports, spy_report, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = reports_to_frame([spy_report, *reports])
    df.to_csv(out / "data_quality.csv", index=False)
    counts = df["status"].value_counts().to_dict()
    return df, counts


def _print_report(picks, meta):
    print(f"\nDAILY PICKS  scan {meta['scan_time_utc']}  |  market bar date {meta['market_bar_date']}  |  "
          f"examined {meta['tickers_examined']}/{meta['tickers_requested']}  excluded(data) {meta['tickers_excluded_data']}")
    if meta.get("data_note"):
        print(f"WARNING: {meta['data_note']}")
    print(f"Data exclusions: {meta['exclusion_reasons_data'] or 'none'}")
    print(f"Examined but not selected: {meta['examined_not_selected_reasons']}")
    print(f"Qualified {meta['tickers_qualified']} tickers; showing {meta['picks_returned']} (max {meta['max_picks']}, min score {meta['min_score']}).")
    if meta.get("verified_only"):
        print(f"--verified-only: dropped qualified-but-unverified tickers by reason: {meta['qualified_but_unverified_dropped'] or 'none'}")
    if picks.empty:
        print("0 candidates passed the rules. That is a valid outcome; nothing is added to fill the list.")
    else:
        show = picks.assign(setups=picks["setups"].map(", ".join), name=picks["company_name"].str.slice(0, 22))
        cols = ["rank", "ticker", "name", "setups", "score", "close", "stop", "target", "shares", "earnings_status", "plan_status"]
        print(show[cols].to_string(index=False))
        for _, r in picks.iterrows():
            print(f"\n#{r['rank']} {r['ticker']}: {r['explanation']}\n   Invalidation: {r['invalidation']}"
                  f"\n   Not fully verified because: {r['unverified_because'] or '-'}")
            if r["news_status"] == "OK":
                print(f"   Headlines (display only): {r['news_headlines']}")
            else:
                print(f"   News: {r['news_status']} (not evidence of no news). Check: {r['news_link']}")
    print("\n" + meta["price_label"] + "\nVERIFIED_PLAN = every mandatory check passed (still NOT a recommendation). "
          "RESEARCH_CANDIDATE = at least one check failed or could not be verified (e.g. earnings date unknown).")
    print(meta["disclaimer"])


def _print_news(n, link=None):
    """Headlines are display-only; an empty feed is NOT_AVAILABLE, never 'no news'."""
    if n["status"] == "OK":
        print("   Recent headlines (display only, not scored):")
        for h in n["headlines"]:
            print(f"     - {h['title']} ({h['publisher']}, {h['published'][:10]})")
    else:
        print(f"   News: {n['status']} - {n['note']}")
    print(f"   Check news yourself: {n['link']}")


def _positive_price(text):
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number") from None
    if not (v > 0 and v < float("inf")):
        raise argparse.ArgumentTypeError("entry price must be a positive number")
    return v


def _print_at_entry(a):
    if not a["available"]:
        print(f"Plan at your entry: unavailable - {a['reason']}"); return
    print(f"Plan at YOUR entry {a['entry']} (price you supplied; gap vs reference close {a['gap_vs_reference_pct']:+.2f}%, {a['gap_atr']:+.2f} ATR):")
    print(f"   - Stop {a['stop']}   Target {a['target']}   Reward/risk {a['rr_gross']} gross, {a['rr_net_of_costs']} net of costs")
    if a["shares"]:
        print(f"   - Size: {a['shares']} share(s) = ${a['position_value']}, risk at stop about ${a['risk_at_stop_usd']}, "
              f"est. round-trip cost ${a['est_round_trip_cost_usd']}")
    else:
        print(f"   - Size: {a['sizing_note']}")
    for w in a["warnings"]:
        print(f"   ! {w}")


def _fmt_check(c):
    return f"{'PASS' if c['passed'] else 'FAIL'}  {c['rule']}" + (f"  [{c['value']}]" if c["value"] is not None else "")


def _run_plan(args, s, spy, tickers, frames, reports, outdir):
    names, rmap = load_names(args.universe_file), {r.ticker: r for r in reports}
    plans = []
    try:
        for t in tickers:
            plans.append(build_plan(t, frames.get(t), spy.frame, rmap.get(t), s, name=names.get(t, ""),
                                    earnings_lookup=next_earnings_date,
                                    news_lookup=None if args.no_news else recent_headlines))
            if args.entry is not None and plans[-1]["plan_status"] != "NO_PLAN":
                plans[-1]["at_entry"] = plan_at_entry(plans[-1], args.entry, s)
    except StaleMarketData as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    for p in plans:
        print(f"\n{'=' * 78}\n{p['ticker']}  {p.get('company_name') or '(name not in universe file)'}  |  completed bar {p['market_bar_date']}")
        if p["plan_status"] == "NO_PLAN":
            print(f"STATUS: {p['status']} - {p['reasons']}"); continue
        if p.get("data_note"):
            print(f"WARNING: {p['data_note']}")
        sc = f"{p['score']:.1f}/100 (min {p['min_score_required']:g})" if p["score"] is not None else "n/a"
        print(f"STATUS: {p['status']}   plan_status: {p['plan_status']}   score: {sc}   close {p['close']:.2f}")
        print(f"Matching setups: {', '.join(p['setups']) or 'none'}")
        if p["unverified_because"]:
            print(f"Not fully verified because: {p['unverified_because']}")
        print("Checks (common + uptrend):")
        for c in p["checklist"]["common"] + p["checklist"]["uptrend"]:
            print("   " + _fmt_check(c))
        for k in ("breakout", "pullback", "relative_strength"):
            state = "MET" if k in p["setups"] else ("disabled in config (enabled_setups)" if k not in s.enabled_setups else "not met")
            print(f"Setup '{k}': {state}")
            for c in p["checklist"][k]:
                print("   " + _fmt_check(c))
        if p.get("explanation"):
            print(f"Setup rules met ({'qualifies' if p['status'] == 'SETUP_QUALIFIED' else 'but NOT a signal: score below minimum'}): {p['explanation']}\nInvalidation: {p['invalidation']}")
        print("Plan:")
        for line in plan_steps(p, s) or ["No levels: indicators incomplete."]:
            print("   - " + line)
        if "at_entry" in p:
            _print_at_entry(p["at_entry"])
        if p["data_warnings"]:
            print(f"Data warnings: {p['data_warnings']}")
        _print_news(p["news"])
    flat = [{**{k: (v if not isinstance(v, (list, dict)) else (";".join(v) if isinstance(v, list) else None))
                for k, v in p.items() if k not in ("checklist", "disclaimer", "at_entry", "news")},
             **{f"at_entry_{k}": (";".join(v) if isinstance(v, list) else v) for k, v in p.get("at_entry", {}).items()}}
            for p in plans]
    pd.DataFrame(flat).to_csv(outdir / "plan.csv", index=False)
    (outdir / "plan.json").write_text(json.dumps({"disclaimer": DISCLAIMER, "plans": plans}, indent=2, default=str))
    print(f"\n{PRICE_LABEL}\n{DISCLAIMER}\nSaved: {outdir/'plan.csv'}  {outdir/'plan.json'}")
    return 0


def main(argv=None, provider=None):
    ap = argparse.ArgumentParser(prog="swing-screener", description="US stock swing screener / research backtester (manual use)")
    ap.add_argument("--config", default="config.toml")
    sub = ap.add_subparsers(dest="command", required=True)
    u = sub.add_parser("universe", help="Build current-listings universe snapshot")
    u.add_argument("--refresh", action="store_true"); u.add_argument("--output", default="data/universe.csv")
    for name, h in (("data", "Download + validate history, write data-quality report"),
                    ("scan", "Manual scan of latest daily bars"), ("backtest", "Sequential walk-forward backtest")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
        p.add_argument("--tickers", help="comma-separated explicit tickers (overrides universe file)")
        p.add_argument("--max-tickers", type=int, default=None); p.add_argument("--universe-file", default="data/universe.csv")
        p.add_argument("--output-dir", default=None)
        if name == "scan":
            p.add_argument("--top", type=int, default=None)
            p.add_argument("--no-news", action="store_true", help="skip headline lookup (display-only feature)")
            p.add_argument("--verified-only", action="store_true",
                           help="show only picks whose mandatory checks ALL pass (known earnings outside window, data OK, >=1 share)")
    pl = sub.add_parser("plan", help="Trading plan for the tickers you type (same rules/data checks as scan)")
    pl.add_argument("symbols", nargs="+", help="one or more tickers, e.g. AAPL NVDA (commas also accepted)")
    pl.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    pl.add_argument("--entry", type=_positive_price, default=None,
                    help="your actual or intended entry price (single ticker only): recomputes stop, target, shares and risk")
    pl.add_argument("--no-news", action="store_true", help="skip headline lookup (display-only feature)")
    pl.add_argument("--universe-file", default="data/universe.csv"); pl.add_argument("--output-dir", default=None)
    sub.add_parser("config", help="Print effective configuration")
    args = ap.parse_args(argv)
    s = load_settings(args.config)
    provider = provider or YFinanceProvider()
    if args.command == "config":
        print(json.dumps({k: v for k, v in s.__dict__.items()}, indent=2, default=list)); return 0
    if args.command == "universe":
        if not args.refresh and Path(args.output).exists():
            print(f"{args.output} exists; use --refresh to rebuild."); return 0
        df = build_universe(args.output)
        print(f"Wrote {len(df)} current-listing tickers to {args.output} (survivorship-biased snapshot)"); return 0
    if args.command == "plan":
        args.tickers = ",".join(args.symbols); args.max_tickers = None
        if args.entry is not None and len({t.strip().upper() for t in args.tickers.split(",") if t.strip()}) != 1:
            print("ERROR: --entry applies to exactly one ticker; run plan separately for each.", file=sys.stderr); return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        spy, tickers, frames, reports = _load(args, s, provider, scan=args.command in ("scan", "plan"))
    except DataProviderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    outdir = Path(args.output_dir or f"reports/{args.command}_{stamp}")
    dq, counts = _write_quality(reports, spy.report, outdir)
    print(f"Data: {len(frames)}/{len(tickers)} tickers usable {counts}. Quality report: {outdir/'data_quality.csv'}")
    if args.command == "data":
        return 0
    if args.command == "plan":
        return _run_plan(args, s, spy, tickers, frames, reports, outdir)
    if not frames:
        print("ERROR: no usable ticker data; nothing to evaluate.", file=sys.stderr); return 2
    if args.command == "scan":
        try:
            picks, meta = build_daily_picks(frames, spy.frame, reports, s, names=load_names(args.universe_file),
                                            earnings_lookup=next_earnings_date, top_n=args.top or s.top_n,
                                            verified_only=args.verified_only,
                                            news_lookup=None if args.no_news else recent_headlines)
        except StaleMarketData as exc:
            print(f"ERROR: {exc}", file=sys.stderr); return 2
        picks.assign(setups=picks["setups"].map("; ".join) if len(picks) else picks.get("setups")).to_csv(outdir / "scan.csv", index=False)
        (outdir / "scan.json").write_text(json.dumps({"meta": meta, "picks": json.loads(picks.to_json(orient="records"))}, indent=2))
        _print_report(picks, meta)
        print(f"Saved: {outdir/'scan.csv'}  {outdir/'scan.json'}")
        return 0
    used = sorted(frames)
    info = snapshot(tickers, used, [r.ticker for r in reports if not r.ok], outdir)
    try:
        summary = run_walk_forward(frames, spy.frame, s, output_dir=outdir, universe_info=info)
    except InsufficientHistory as exc:
        print(f"ERROR: insufficient history for a walk-forward run: {exc}", file=sys.stderr); return 2
    c, b = summary["strategy_combined_oos"], summary["spy_buy_and_hold_oos"]
    print(f"OOS {summary['oos_period']['start']}..{summary['oos_period']['end']}  windows={summary['oos_period']['n_windows']}  "
          f"trades={c['trades']}\n  strategy {c['total_return_pct']:.2f}%  maxDD {c['max_drawdown_pct']:.2f}%   SPY {b['total_return_pct']:.2f}%  maxDD {b['max_drawdown_pct']:.2f}%")
    print(f"Reports in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
