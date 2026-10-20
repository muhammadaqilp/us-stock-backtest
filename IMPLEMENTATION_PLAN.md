# Implementation Plan — US Swing Screener

## User-selected settings
- Universe: broad liquid US-listed equities, sourced from current Nasdaq Trader symbol directories.
- Strategies: breakout momentum, trend pullback, relative-strength continuation.
- Holding horizon: 5–15 trading sessions; stops/targets may close a trade earlier.
- Capital: USD 1,000.
- Interface: command-line interface (CLI).
- First milestone: historical walk-forward research/backtest before relying on daily picks.
- Data: yfinance daily OHLCV, adjusted prices (`auto_adjust=True`).

## Milestones

### Phase 1 — Data foundation
1. Fetch current Nasdaq Trader listed-security directories.
2. Normalize tickers and apply heuristic filters for ETFs/test issues/non-common instruments.
3. Download daily OHLCV and cache per ticker.
4. Validate required columns, positive prices, high/low consistency, sorted unique dates, volume, and minimum history.
5. Persist a ticker-level data quality report; skip failed symbols rather than fabricating values.

**Acceptance:** a reproducible universe snapshot, explicit failed-ticker report, and no silent filling of missing OHLCV.

### Phase 2 — Indicators and setup scoring
1. Calculate SMA20/SMA50/SMA200, 20/60-day returns, prior 20-day high, relative volume, ATR14, average dollar volume, and 60-day relative strength versus SPY.
2. Require price >= $5, average dollar volume >= $20m, positive trend, valid ATR, and acceptable volatility.
3. Score candidates using fixed, documented weights: trend 25%, momentum 25%, relative strength 20%, volume 15%, risk/volatility 15%.
4. Emit a row for each qualifying setup with its score, indicators, proposed stop, target, and planned reward/risk.
5. Keep score weights fixed until a research protocol is defined; do not tune against the final test period.

**Acceptance:** deterministic results for the same input data/configuration; no look-ahead in the prior-high feature.

### Phase 3 — Backtest and risk model
1. Use an initial training/warm-up window of at least two years and evaluate later dates out of sample.
2. Generate signals using data through day *t* close; enter no earlier than day *t+1* open.
3. Model stop at 2 ATR and target at 4 ATR; max hold 15 sessions.
4. Resolve ambiguous daily candles pessimistically (stop first); model gaps and estimated transaction costs.
5. Size positions with whole shares, 0.5% of equity risk budget, 20% max equity per position, cash limits, and max five simultaneous positions.
6. Save trade ledger, daily equity curve, drawdown, win rate, profit factor, and explicit caveats.

**Acceptance:** test coverage for indicators/data validation; reproducible trade ledger; no use of future bars to create an entry signal; results include costs and do not imply guaranteed returns.

### Phase 4 — CLI and reports
- `universe --refresh`: refresh ticker snapshot.
- `data`: download/validate history.
- `backtest`: run historical simulation and save JSON/CSV reports.
- `scan`: rank the latest available candidates and save a dated CSV.
- `config`: print effective parameters.

**Acceptance:** commands fail loudly on critical data problems and clearly report zero qualifying candidates.

### Phase 5 — Daily automation (DEFERRED: explicitly out of scope for the current phase; nothing implemented)
1. Run after the US regular session closes, allowing the Yahoo daily bar to settle.
2. Use a timezone-aware scheduler configured for `America/New_York` or a wrapper that accounts for daylight saving time and market holidays.
3. Log stdout/stderr, timestamp reports, and monitor stale data, missing SPY data, rate limits, and repeated failures.
4. Keep the first version alert/report-only; no brokerage order placement.

**Acceptance:** daily run creates a report or a clear failure, never silently reuses obviously insufficient history, and is observable through logs.

## Known research gaps before real-money use
- Current listing membership creates survivorship bias. Add point-in-time historical constituents and delisted stocks.
- yfinance data quality/availability and adjustment conventions need independent checks.
- Add point-in-time earnings/event calendar, sector exposure controls, benchmark comparison, and sensitivity analysis.
- Test multiple realistic cost assumptions and compare simulated versus paper-trading fills.
- Report results by walk-forward window, regime, setup, and liquidity bucket; include uncertainty and avoid overfitting.
- Paper trade for multiple months before considering live use. No strategy is presumed profitable.
