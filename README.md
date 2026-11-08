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
swing-screener plan HAS --entry 111.50                   # recompute stop/target/shares/risk from YOUR fill price (one ticker)
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

### Firm plan from your fill: `plan TICKER --entry PRICE`

After the open you know your real price. `swing-screener plan HAS --entry 111.50` keeps the signal-date ATR14 and recomputes, from *your* price: stop (entry - 2 ATR), target (entry + 4 ATR), gross/net reward-risk, whole-share size under the same 0.5%-risk / 20%-cap / cash limits, and costs. It also prints the gap versus the reference close (in % and ATR) and warnings: implausible price (> 25% from the reference: typo / wrong currency / wrong ticker), buying after a gap above 1 ATR, above the optional max entry price, setup not qualified, earnings unknown or inside the holding window. If the risk budget cannot buy one share it says so and never raises risk. The entry is a number you supply; the tool does not know live prices and does not change the plan's verification status. One ticker per call; results are also in `plan.json` (`at_entry`) and `plan.csv` (`at_entry_*`).

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

## Rule variants - pre-registered 2026-11-04, before any holdout run

Goal: a higher win rate only counts if expectancy per trade and profit factor do not get worse. Win rate alone is easy to
raise (a closer target wins more often but earns less per win), so it is reported but never the deciding number.

Holdout: **2011-01-01 to 2021-11-02** (2-year warm-up, so out-of-sample 2013 to 2021), same seeded 400-ticker sample, same
costs, separate cache (`data/cache_holdout_2011`). No rule in this project was designed or tuned on this period. It includes
the 2015-16 drawdown, Q4 2018 and the 2020 crash, so it is not only a bull market. Survivorship bias is worse here than
in the recent period (today's listings looked back 10+ years).

| Variant | Change (config) | Where the idea came from |
|---|---|---|
| V1 regime | `regime_sma = 200`: no new entries unless SPY closed above its 200-day SMA on the signal day | Standard trend-filter idea, not from this project's data |
| V2 score | `min_score = 75` | Score quartiles in the 2023-26 backtest (so only the holdout can test it) |
| V3 no RS | `enabled_setups = ["breakout", "pullback"]` | Relative-strength setup lost money 2023-26 (so only the holdout can test it) |
| **V4 = V1+V2+V3** | **PRIMARY hypothesis** | |
| T1 target 2 ATR | `atr_target_multiple = 2.0` | **Illustration only**: shows win rate rising without expectancy improving. Never a candidate |

**Decision rule (fixed in advance):** change the defaults to V4 only if, over the holdout, V4 beats the baseline on BOTH
expectancy per trade (%) and profit factor, has profit factor > 1 at 2x costs, and has at least 100 trades. Otherwise the
defaults stay as they are. V1-V3 are context; none is adopted on its own result. The 2023-26 period is also reported but
cannot count as evidence for V2/V3, which were derived from it.

### Result (`reports/variants_20261104/`, reproduce with `python research/run_variants.py`)

Holdout, out-of-sample 2013-01-03 to 2021-11-01, 268 usable tickers (132 had no or too little history that far back).
SPY buy and hold over the same dates: **+214%**. Expectancy is the mean per-trade return after costs, with a 90% bootstrap interval.

| Variant | Trades | Win rate | Expectancy/trade | Profit factor | PF at 2x costs | Return | Max DD |
|---|---|---|---|---|---|---|---|
| baseline | 943 | 46.4% | +0.22% [-0.14, +0.60] | 0.98 | 0.88 | -3.0% | -16.5% |
| V1 regime | 882 | 47.1% | +0.26% [-0.13, +0.64] | 0.99 | 0.89 | -2.4% | -15.4% |
| V2 score 75 | 631 | 44.1% | -0.08% [-0.55, +0.36] | 0.92 | 0.84 | -9.6% | -22.1% |
| V3 no RS | 851 | 47.6% | +0.65% [+0.22, +1.08] | 1.13 | 0.88 | +20.5% | -14.4% |
| **V4 primary** | 501 | 44.3% | +0.34% [-0.24, +0.92] | 1.04 | **0.80** | +3.3% | -15.4% |
| T1 target 2 ATR (illustration) | 1244 | **52.9%** | +0.08% [-0.17, +0.34] | 0.99 | 0.86 | -3.1% | -17.2% |

Recent period (2023-11-06 to 2026-11-02, context only; SPY +72%): baseline PF 1.09 / +7.2%; V1 1.11 / +8.2%;
V2 0.99 / -0.6%; V3 1.10 / +7.1%; V4 1.30 / +14.5% (PF 1.24 at 2x costs); T1 0.87 / -11.3% with a 48.5% win rate.

**Decision (by the rule fixed above): V4 is NOT adopted; defaults are unchanged.** It beat the baseline on expectancy
and profit factor but failed the cost test (PF 0.80 at 2x costs). Other findings, none adopted:
- The baseline lost money over 2013-21 (PF 0.98) while SPY tripled. The "no demonstrated edge" conclusion now covers
  about 12 years and two periods.
- V2: a higher minimum score made results **worse** in the holdout. The score-quartile pattern seen in 2023-26 did not
  hold, so there is no evidence the score ranks well.
- V3: dropping the relative-strength setup was the only change with a 90% interval above zero, and RS was the weakest
  setup in both periods (holdout -0.05%/trade over 474 trades). It still fails at 2x costs. It is the best-supported
  next hypothesis for a new pre-registered test on data not yet seen (e.g. forward paper trading), not a proven edge.
- V1: the SPY 200-day filter changed almost nothing.
- T1: the closer target produced the highest win rate in both periods and the worst or equal expectancy. Win rate is
  the wrong number to optimise.

## Breakeven stop - pre-registered 2026-11-04, before any run on its test data

Idea from trade diagnostics on the two periods above (so neither can test it): about 40% of losing trades were up at
least 1 ATR before falling to the full stop.

Rule `breakeven_trigger_atr = 1.0` (**B1, primary**): once a session's high reaches `entry open + 1.0 x ATR14` (the
signal-day ATR used for the stop), the stop is raised from the **next** session to the breakeven price
`entry x (1 + cost) / (1 - cost)`, where a stop exit nets roughly zero after both sides' costs. Daily bars do not show
whether the high came before the low, so the raised stop never applies to the bar that triggered it. The stop is never
lowered; gap-through-stop still fills at the open; target and time exits are unchanged.

Secondary hypothesis: **V3** (`enabled_setups = ["breakout", "pullback"]`), whose fresh test this is. Context only:
B1 + V3 together.

Test data: **2001-01-01 to 2012-12-31** (2-year warm-up, so out-of-sample 2003 to 2012, including 2008), same seeded
400-ticker sample and costs, separate cache (`data/cache_holdout_2001`). Never used before in this project. Survivorship
bias is worst here: companies that failed in 2008 are absent.

**Decision rule (same as before, applied to B1 and to V3 separately):** adopt as the default only if, over 2003-2012, it
beats the baseline on BOTH expectancy per trade (%) and profit factor, has profit factor > 1 at 2x costs, and has at
least 100 trades. Win rate is reported but does not decide. B1 + V3 is never adopted on its own result.

### Result (`reports/breakeven_20261104/`, reproduce with `python research/run_variants.py --set breakeven --period all`)

Out-of-sample 2003-01-02 to 2012-12-31, 183 usable tickers (217 had no or too little history that far back). SPY buy and
hold over the same dates: +60%. A breakeven exit nets ~0 after costs and counts as **not** a win.

| Variant | Trades | Win rate | Breakeven exits | Expectancy/trade | Profit factor | PF at 2x costs | Return | Max DD |
|---|---|---|---|---|---|---|---|---|
| baseline | 982 | 45.7% | - | +0.09% [-0.27, +0.46] | 0.99 | 0.90 | -2.2% | -23.3% |
| **B1 breakeven 1 ATR** | 1257 | **25.5%** | 486 | -0.13% [-0.40, +0.15] | **0.89** | 0.79 | -19.8% | -29.2% |
| **V3 no RS** | 886 | 46.8% | - | +0.25% [-0.12, +0.66] | 1.07 | **0.96** | +12.9% | -19.6% |
| B1 + V3 (context) | 1054 | 25.0% | 436 | -0.05% [-0.35, +0.26] | 0.91 | 0.79 | -13.0% | -19.3% |

The two already-used periods (context only) agree: B1 lowers win rate to about 26% in both, and lowers profit factor
in 2013-21 (0.95 vs 0.98). In 2023-26 it is slightly higher (1.12 vs 1.07) but no better at 2x costs. The recent-period run had 347
usable tickers (367 earlier the same day; some downloads failed), so its baseline differs slightly from the table above.

**Decision (by the rule fixed above): neither is adopted; defaults are unchanged.**
- **B1 is rejected.** It made expectancy and profit factor worse and cut the win rate roughly in half. Raising the
  stop to breakeven cuts off trades that dip after +1 ATR and would have recovered: many of the small
  time-exit winners become ~0 exits, and they are not replaced by enough saved losers.
- **V3 failed the cost test again** (PF 0.96 at 2x costs). It is now better than the baseline on expectancy and
  profit factor in all three periods (2003-12, 2013-21, 2023-26), but the improvement is small, its 90% interval
  includes zero in two of them, and it does not survive doubled costs in either holdout. Consistent but not enough to
  call an edge.
- The baseline lost money in 2003-12 as well (PF 0.99). Across about 22 years of tests it has not shown an edge.

## Limitations

Today's-listings universe (survivorship + look-ahead in membership); sample of 400, not the full listing; Yahoo data can be revised/incomplete; daily bars cannot show intraday order, spreads, partial fills or halts; no earnings filter in the backtest (earnings gaps are inside the results); dividends ignored for both sides; the strategy is long-only with a single 4-year period (one regime); costs are assumptions; $1,000 with whole shares skips many signals (597 zero-share skips). `signals.py` (an old, defective prototype) was removed.

## Tests

`python -m pytest -q` — 158 tests, all on **synthetic** fixtures (`tests/synthetic.py`); they verify mechanics and are not market results. Real-data outputs exist only under `reports/`.
