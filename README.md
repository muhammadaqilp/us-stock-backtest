# US Swing Screener (research tool)

Manual CLI for screening liquid US equities with three swing setups and evaluating them with a sequential walk-forward
backtest. **Research only.** No broker connection, no orders, no scheduling or notifications. Nothing here is a
recommendation, and the current evidence (below) does not support claiming an edge.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

## Commands (all manual)

```bash
swing-screener config                                   # effective configuration
swing-screener universe --refresh                       # rebuild current-listings snapshot (Nasdaq Trader)
swing-screener data     --max-tickers 400               # download + validate, write data_quality.csv
swing-screener scan     --max-tickers 400               # DAILY PICKS: up to 10 candidates (CSV + JSON), latest completed session
swing-screener backtest --max-tickers 400               # sequential walk-forward (CSV + JSON)
swing-screener plan AAPL NVDA SU                         # trading plan for tickers YOU type (any stock, qualified or not)
# common flags: --refresh (ignore cache), --tickers AAPL,MSFT, --output-dir DIR, --universe-file FILE, --config FILE
```

Outputs go to `reports/<command>_<timestamp>/` (or `--output-dir`): `data_quality.csv`, `scan.csv/json`,
`trades.csv`, `equity.csv`, `windows.csv`, `cost_sensitivity.csv`, `entry_filter.csv`, `skipped_entries.csv`, `summary.json`,
`universe_snapshot.csv/json` (exact ticker set + sha256 per run). `--max-tickers` draws a seeded random sample
(`sample_seed`) so it is reproducible and not alphabetically biased.

## Data layer and the yfinance MCP — verified findings (2026-10-17)

| Item | Finding |
|---|---|
| MCP tool for history | `get_historical_stock_prices(ticker, period, interval)` only. Docstring mentions start/end but **no such parameters exist**; history is window-from-today (`period`: up to `max`). |
| Fields returned | Date (UTC 04:00Z), Open, High, Low, Close, Volume, Dividends, Stock Splits, Capital Gains. **No `Adj Close`** despite the docstring. |
| Adjustment | **OHLC are dividend-adjusted** (auto_adjust). Checked: SPY 2026-07-17 MCP Close 741.4488 = Yahoo AdjClose in our cache; raw close was 743.29. These are *not* executable prices. |
| Reachability | Agent-side tool only; the CLI cannot call MCP. Output is a large JSON string (~16 KB for 3 months), impractical for 5y × hundreds of tickers. |

Decision: the production provider is the **yfinance Python library with `auto_adjust=False`** (same Yahoo source). OHLCV are
split-adjusted, **not** dividend-adjusted (approximately what traded); `AdjClose` is stored for reference only and never used
for fills. `McpExportProvider` can ingest a saved MCP JSON export for cross-checks; it tags the data
`mcp_fully_adjusted_ohlc…`, raises a "NOT EXECUTABLE PRICES" warning, and its cache never mixes with executable data.

Validation (never modifies or fills data): required columns, numeric, NaN/inf, positive prices, high/low vs open/close, negative
volume, duplicate and unsorted dates, minimum history, missing sessions vs SPY calendar (> `max_missing_fraction` ⇒ FAILED, fewer ⇒ warning, not
filled), stale last bar (`max_stale_sessions`), zero-volume share, >50% daily moves. Every ticker gets a report with source,
retrieval time, date range, rows, adjustment convention, status. Cache: `data/cache/<T>.csv` + `.meta.json`; `--refresh`
forces re-download; scan re-fetches data older than `scan_cache_max_age_hours` and refuses to run on stale SPY data. Exchange holidays are not modelled
(tolerance via `max_stale_sessions`).

**Survivorship bias:** the universe is *today's* Nasdaq Trader listing (ETFs/test issues/warrants etc. filtered by name heuristics).
Delisted/bankrupt names are absent and membership uses present-day knowledge, so historical results are biased upward by an unknown amount.
Point-in-time constituents need a paid/other data source (not available here).

## Strategy rules (all thresholds in `config.toml`; hypotheses, not tuned)

Common filters: all indicators finite, close ≥ $5, 20-day average dollar volume ≥ $20M, 0 < ATR%(14) ≤ 10%.
Established uptrend: **Close > SMA50 > SMA200**. Qualification (`strategies.py`) is separate from ranking (`scoring.py`).

1. **Breakout momentum** — uptrend; close > highest high of the *previous* 20 sessions (current bar excluded via shift); relative volume ≥ 1.5 (vs the prior 20-session mean).
   *Rationale:* a close through a recent range on above-normal participation is a standard momentum-continuation trigger.
2. **Trend pullback** — uptrend; within the last 5 sessions the low came within 1% of SMA20 and the close was ≥ 3% below the 10-session high; **reversal confirmation today:** close > open, close > prior day's high, close > SMA20. Proximity alone does not qualify.
3. **Relative-strength continuation** — uptrend; 20-day return > 0; stock return beats SPY by > 0 over *both* 20 and 60 sessions; continuation: close within 5% of the 60-session high, above SMA20, relative volume ≥ 1.0.

A name can match several setups; the scan consolidates them into ONE row per ticker (all setups and explanations kept), the backtester takes the best-scoring one per ticker.

**Score (0–100, ranking only, not a probability of profit):** trend 25 (close>SMA50 8, SMA50>SMA200 8, SMA50 20-day slope 0–5% → 9), momentum 25 (60% 20d return 0–15%, 40% 60d return 0–30%), relative strength 20 (mean of 20d/60d excess vs SPY, −5%…+15%), volume 15 (rel. volume 1–2.5), risk 15 (ATR% 2–8%, calmer scores higher). Minimum score 60 and gross R/R ≥ 2.

## Daily stock-pick report (`scan`)

Manual research aid; run it after the US close. It never trades, schedules or notifies.

1. SPY and every ticker are validated; bars from a session that is not yet complete (after `expected_last_session`, 16:30 ET cutoff) are dropped; refuses to run if SPY's last completed bar is stale.
2. Tickers whose latest bar is not SPY's latest completed bar, or whose latest bar has zero volume, are excluded; invalid / insufficient-history tickers are excluded and counted by reason.
3. Each remaining ticker is tested against the three independent setups, scored (0-100), filtered by `min_score` (60), and consolidated to one row per ticker. **Up to 10 (`top_n`) are returned; fewer or zero if fewer qualify - the list is never padded.**

Per pick: rank, ticker, company name (from the universe snapshot), primary setup + all `setups`, score + components, signal date, latest completed bar date, close, SMA20/50/200, 20/60d momentum, RS vs SPY, relative volume, ATR14/ATR%, reference entry (signal-date close), stop (2 ATR), target (4 ATR), gross and net R/R, shares for $1,000 under the 0.5%-risk / 20%-cap / cash limits, estimated round-trip cost, earnings date + `earnings_status` (`IN_HOLDING_WINDOW` / `OUTSIDE_HOLDING_WINDOW` / `UNKNOWN`), data warnings, a plain-language `explanation` per setup and an `invalidation` note. **All prices are reference levels derived from the signal date, not guaranteed execution prices**; the backtest convention is a next-session-open entry with stop/target re-based on that open. Invalidation notes other than the stop are manual guidance and are not backtested.

`plan_status = VERIFIED_PLAN` only if every mandatory check passes: data validated and not stale, earnings date known and outside the holding window, >= 1 share fits, net R/R > 0. Otherwise `RESEARCH_CANDIDATE` with `unverified_because` (e.g. `earnings_unverified`, `earnings_in_holding_window`, `zero_shares(risk_budget)`). Earnings dates come from yfinance `Ticker.calendar` (forward-looking, scan only); a missing or past date is `UNKNOWN`, never assumed clear. Even a `VERIFIED_PLAN` is not a recommendation: the walk-forward results below show no demonstrated edge.

Files: `scan.csv` (setups joined with `;`), `scan.json` (`{"meta": {...}, "picks": [...]}` - meta has scan time, market-bar date, tickers requested/examined/excluded, exclusion reasons, reasons examined tickers were not selected), `data_quality.csv`. The terminal shows the same header, a table and per-pick reasons.

## Trading plan for tickers you type (`plan`)

```bash
swing-screener plan AAPL NVDA SU          # lower/upper case and commas both fine; add --refresh for fresh data
```

Uses the same data validation, completed-bar rule, rules, score, levels, sizing and earnings check as `scan`, for any ticker (it does not have to be in the 400-stock sample). For each ticker it prints: `status` (`SETUP_QUALIFIED`, `SETUP_BELOW_MIN_SCORE`, `NO_SETUP_TODAY`, `DATA_NOT_CURRENT`, `DATA_UNAVAILABLE`), the score, a PASS/FAIL checklist of every rule for all three setups (so you see exactly what is missing), and plan steps (reference entry, stop, target, reward/risk, time exit, position size for $1,000, earnings). Levels for a ticker that does not qualify are labelled **HYPOTHETICAL / NOT A SIGNAL**; if a ticker has no current completed bar or invalid data, no plan is produced. Files: `plan.csv`, `plan.json`, `data_quality.csv`. A test asserts the checklist agrees with the real qualifiers.

## Backtest model

Signals use data through the signal-day close; entry is the **next session's open**; indicators are causal (truncation-invariance tested). Per-side cost = 5 bps commission + 10 bps slippage on entry and every exit (including stop/target/time/final liquidation). Sizing: whole shares, min of 0.5% equity risk (risk per share includes costs), 20% equity cap, available cash; max 5 positions, one per ticker; **zero shares ⇒ skip (risk is never raised)**. Exits: open gap through stop/target fills at the open; otherwise stop checked before target (pessimistic when a bar touches both); max 15 sessions ⇒ close. Missing bars are never filled: the last close is carried and counted; a missing entry bar skips the trade. Final session: every open position is liquidated at the close with costs *before* the final equity row, and a ledger invariant (final equity = initial + Σ trade P&L, no open positions) is asserted on every simulation.

Walk-forward: 2-year warm-up (indicators only; **no parameter is fitted on any data**), then sequential non-overlapping 3-month out-of-sample windows in one continuous account (open positions carry across boundaries; window figures are slices by exit date). Reported per window and combined: return, annualized return, max drawdown, trades, win rate, avg win/loss, profit factor, expectancy after costs, exposure, turnover, costs, per-setup breakdown, SPY buy-and-hold over the same dates (same per-side costs, price only, no dividends), and cost sensitivity at 1×/2×/3×. A final partial window is flagged.

## Actual results (real data, produced 2026-10-17; `reports/backtest_20261017/`)

Sample: seeded 400 tickers of today's listings (371 passed validation; 29 failed on < 260 rows). OOS 2023-10-18 → 2026-10-16, 12 windows, 370 trades.

| | Strategy | SPY buy & hold |
|---|---|---|
| Total return | **+5.83%** (≈ +1.9%/yr) | **+75.26%** (≈ +20.6%/yr) |
| Max drawdown | −8.79% | −19.00% |
| Win rate / PF / expectancy | 43.8% / 1.07 / +$0.16 per trade | — |
| Costs paid | $23.1 commission + $46.3 slippage | — |

Cost sensitivity: 2× costs ⇒ −1.86%, 3× ⇒ −4.91% (profit factor 0.98 / 0.93). By setup: breakout +$88 (115 trades), pullback +$6 (81), relative-strength −$36 (174). 5 of 12 windows lost money.

**Conclusion (supported by these numbers only):** no demonstrated edge. The strategy captured ~8% of SPY's return with a smaller drawdown and ~26% average exposure; the small positive result disappears at 2× costs and is within noise for 370 trades. A bull-market, survivorship-biased sample flatters stock-picking results, so the true picture is probably no better. Do not trade these rules.

## Optional entry filter ("do not buy above") - tested, not recommended

Setting `max_entry_gap_atr = k` in `config.toml` (default 0 = off) skips a trade when the next open is above `signal close + k x ATR14`; `scan`/`plan` then print the `max_entry_price`. The walk-forward report compares the baseline against k = 0.25 / 0.5 / 1.0 (`entry_filter.csv`, `summary.json`). **k = 0.5 was fixed as the single primary hypothesis before running**; the others are context only.

Result (`reports/backtest_entryfilter_20261021/`, same data/costs/signals for every row; OOS 2023-10-23 to 2026-10-20, 12 windows; the data had rolled forward 3 days since the 2026-10-17 report, so its baseline differs from the figures above):

| k (ATR) | Return | Max DD | Trades | Profit factor | Expectancy/trade |
|---|---|---|---|---|---|
| baseline (off) | +7.22% | -9.02% | 369 | 1.09 | $0.196 |
| 0.25 | +6.58% | -7.73% | 357 | 1.09 | $0.184 |
| **0.5 (primary)** | **+4.36%** | -8.21% | 361 | 1.06 | $0.121 |
| 1.0 | +6.65% | -8.10% | 366 | 1.08 | $0.182 |

SPY buy and hold over the same dates: +82.73%. **Conclusion: no evidence the filter helps.** The pre-registered k = 0.5 returned less than the baseline; the other values are within noise of it (hundreds of trades, ~$0.1-0.2 expectancy) and slightly lower drawdowns are not enough to conclude anything. Choosing the best-looking k now would be tuning on the test period, so none is recommended. `skipped_for_gap` counts skipped candidates (several candidates can compete for each open slot), not lost trades.

## Limitations

Today's-listings universe (survivorship + look-ahead in membership); sample of 400, not the full listing; Yahoo data can be revised/incomplete; daily bars cannot show intraday order, spreads, partial fills or halts; no earnings filter in the backtest (earnings gaps are inside the results); dividends ignored for both sides; the strategy is long-only with a single 4-year period (one regime); costs are assumptions; $1,000 with whole shares skips many signals (597 zero-share skips). `signals.py` (an old, defective prototype) was removed.

## Tests

`python -m pytest -q` — 124 tests, all on **synthetic** fixtures (`tests/synthetic.py`); they verify mechanics and are not market results. Real-data outputs exist only under `reports/`.
