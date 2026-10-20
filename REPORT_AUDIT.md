# Report Audit — 2026-10-17

Read-only inspection of `reports/data_quality_20261017/`, `reports/backtest_20261017/`, `reports/scan_20261017/`. No report file was modified or regenerated. All figures come from the named files.

## Summary verdicts

| Category | Verdict |
|---|---|
| Data quality (`data_quality_20261017/`) | **Valid** |
| Backtest (`backtest_20261017/`) | **Partially valid** |
| Scan (`scan_20261017/`) | **Valid** |

No errors or incomplete calculations were found. The evidence does **not** show a profitable strategy.

---

## 1. `reports/data_quality_20261017/`

### Files
| File | Purpose | Rows |
|---|---|---|
| `data_quality.csv` | Per-ticker validation report (SPY + 400 sampled tickers) | 401 |

Columns: ticker, source, retrieved_at, start, end, rows, adjustment, status, stale, lag_sessions, missing_sessions, errors, warnings.

### Contents
- Status: **258 OK, 114 WARN, 29 FAILED**.
- Failures: 28 × insufficient history (< 260 rows); 1 × SVA (1 row, non-positive price, inconsistent high/low).
- Warnings: 48 zero-volume-day warnings, 88 daily moves > 50% (a row can have both).
- Date range: all start 2021-10-18; ends 2026-10-14 to 2026-10-16 (only SVA, FAILED, ends before 10-16).
- `missing_sessions` sums to 0.
- Adjustment (all rows): `split_adjusted_not_dividend_adjusted; AdjClose=split+dividend (reference only)`.

### Provenance
Real yfinance (Python library) data. **Not** MCP output, **not** synthetic. 275 rows source = `yfinance (cache)`, 126 = `yfinance` (live). Retrieval 2026-10-17 08:44–08:50 UTC, after the 2026-10-16 close.

### Issues
- `stale` = 0 for every row, but staleness checking was off for the `data` command, so this is not evidence of freshness (`lag_sessions` not evaluated).
- WARN rows are mostly illiquid micro-caps; the backtest liquidity filter excludes them, but they remain in the report.

**Verdict: valid.** Failures are explicit; nothing was filled.

---

## 2. `reports/backtest_20261017/`

### Files
| File | Purpose | Rows |
|---|---|---|
| `summary.json` | Metrics, settings, caveats, cost sensitivity, windows, per-setup results | — |
| `trades.csv` | Trade ledger | 370 |
| `equity.csv` | Daily equity (incl. pre-liquidation equity) | 752 |
| `windows.csv` | Out-of-sample window results | 12 |
| `cost_sensitivity.csv` | 1× / 2× / 3× cost scenarios | 3 |
| `skipped_entries.csv` | Signals not taken, with reason | 5,157 |
| `universe_snapshot.csv` / `.json` | Exact ticker set (400 requested, 371 used) + hash | 400 |
| `data_quality.csv` | Same schema as folder 1 (later run; timestamps differ) | 401 |

`summary.json` status flag: `RESEARCH_ONLY_NOT_VALIDATED`.

### Key metrics (`summary.json`, `windows.csv`, `cost_sensitivity.csv`)
| | Strategy | SPY buy & hold |
|---|---|---|
| Total return | +5.83% | +75.26% |
| Max drawdown | −8.79% | −19.00% |

- OOS: 2023-10-18 → 2026-10-16, 12 windows (last partial), 370 trades, 112 tickers.
- Cost sensitivity: 2× costs → −1.86%; 3× → −4.91%.
- Skips: 4,560 `max_positions`, 597 `zero_shares`.

### Provenance
Real yfinance data (cached + fresh). Not synthetic, not MCP.

### Verification checklist
| Check | Result | Evidence |
|---|---|---|
| Valid historical OHLCV | Yes — 371/400 validated; 29 excluded, not filled; raw-style split-adjusted prices | `universe_snapshot.csv`, `data_quality.csv` |
| Next-session entry | Yes — every `entry_date` is strictly after `signal_date` | `trades.csv` |
| Realistic costs | Yes — exactly 15 bps entry and 15 bps exit on all trades; commission/slippage tracked; 2×/3× scenarios present. Assumed, not broker quotes | `trades.csv`, `cost_sensitivity.csv` |
| Genuinely out-of-sample | Yes — first OOS date 2023-10-18 (2-year warm-up), sequential non-overlapping windows, no parameters fitted. Universe membership is still today's listings (look-ahead) | `summary.json`, `windows.csv` |
| Accounting | Initial capital + Σ trade P&L = final equity; 0 open positions at end; 5 `end_of_test` closes; max 5 concurrent positions; no trade > 15 sessions; no NaNs; max exposure 48% | `equity.csv`, `trades.csv` |
| Exits | 37 of 370 were gap exits; 0 ambiguous bars; 0 missing-bar days | `trades.csv` |

### Issues
- Sample is 400 tickers, not the full listing.
- Survivorship bias (today's listings); single bull-market regime.
- SPY comparison is price-only (no dividends).

**Verdict: partially valid.** Mechanics and OOS separation are sound; results are biased and show no edge.

---

## 3. `reports/scan_20261017/`

### Files
| File | Purpose | Rows |
|---|---|---|
| `scan.csv` | Ranked candidates with scores, levels, sizing, costs, event risk | 11 (36 columns) |
| `scan.json` | Same content as `scan.csv` | 11 |
| `data_quality.csv` | Validation report for the scan run | 401 |

### Contents
- All rows: signal date **2026-10-16**, latest bar date **2026-10-16**.
- `data_quality.csv`: 258 OK / 114 WARN / 29 FAILED; `stale` = 0 (check was on for the scan).
- Plan status: 10 `RESEARCH_CANDIDATE`, 1 `EXECUTABLE_PLAN` (DEO).
  - Most downgraded for `earnings_in_holding_window`; MANH, COR, ADP also `zero_shares(risk_budget)`.
  - DEO: earnings date unconfirmed (NOT_CHECKED) yet still labelled executable.
- `data_warnings` empty for all 11 rows.

### Provenance
Real yfinance prices (cache through 2026-10-16) plus real forward earnings dates from the yfinance calendar. Not synthetic, not MCP.

### Freshness
Based on the latest completed session (Friday 2026-10-16 close); files generated Saturday 2026-10-17. **Not stale.**

### Issues
- DEO is labelled executable despite unchecked earnings risk; treat as a research candidate.
- SU and VRNS each appear twice (one row per setup).
- Score is a ranking, not a probability of profit.

**Verdict: valid** as a snapshot of the 2026-10-16 close.
