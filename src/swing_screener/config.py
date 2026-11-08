"""Configuration loading. config.toml sections map onto flat Settings fields; unknown keys are rejected."""
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # data
    source: str = "yfinance"
    history_years: int = 5
    cache_dir: str = "data/cache"
    pause_seconds: float = 0.3
    max_stale_sessions: int = 1
    max_missing_fraction: float = 0.02
    min_history_rows: int = 260
    max_abs_daily_move: float = 0.50
    scan_cache_max_age_hours: float = 12.0
    # universe
    min_price: float = 5.0
    min_avg_dollar_volume: float = 20_000_000
    sample_seed: int = 42
    max_tickers: int = 150
    # strategy
    max_atr_pct: float = 0.10
    min_reward_risk: float = 2.0
    min_score: float = 60.0
    top_n: int = 10
    atr_stop_multiple: float = 2.0
    atr_target_multiple: float = 4.0
    max_holding_days: int = 15
    breakeven_trigger_atr: float = 0.0      # 0 = off; else once high >= entry + k*ATR, stop -> breakeven from the NEXT session
    max_entry_gap_atr: float = 0.0          # 0 = disabled; else skip entry if next open > signal close + k*ATR
    entry_filter_tests: tuple = (0.25, 0.5, 1.0)   # k values compared in the walk-forward report (primary hypothesis: 0.5)
    breakout_lookback: int = 20
    breakout_volume_multiple: float = 1.5
    pullback_ma_window: int = 20
    pullback_tolerance: float = 0.01
    pullback_lookback: int = 5
    pullback_high_window: int = 10
    pullback_min_depth: float = 0.03
    rs_lookbacks: tuple = (20, 60)
    rs_min_outperformance: float = 0.0
    rs_high_window: int = 60
    rs_max_distance_from_high: float = 0.05
    general_relative_volume: float = 1.0
    regime_sma: int = 0                     # 0 = off; else no new entries unless SPY closed above its N-day SMA on the signal day
    enabled_setups: tuple = ("breakout", "pullback", "relative_strength")
    # risk
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.20
    max_positions: int = 5
    # costs
    commission_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 10.0
    cost_sensitivity_multipliers: tuple = (1.0, 2.0, 3.0)
    # walk-forward
    warmup_years: int = 2
    test_months: int = 3

    @property
    def cost_rate(self) -> float:
        """Total per-side cost as a fraction of price (commission + slippage)."""
        return (self.commission_bps_per_side + self.slippage_bps_per_side) / 10_000.0

    def with_cost_multiplier(self, m: float) -> "Settings":
        return replace(self, commission_bps_per_side=self.commission_bps_per_side * m,
                       slippage_bps_per_side=self.slippage_bps_per_side * m)


def load_settings(path: str | Path = "config.toml") -> Settings:
    p = Path(path)
    if not p.exists():
        return Settings()
    with p.open("rb") as f:
        raw = tomllib.load(f)
    known = {f.name: f for f in fields(Settings)}
    values = {}
    for section, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"config: top-level key '{section}' must be a section")
        for key, val in body.items():
            if key not in known:
                raise ValueError(f"config: unknown key '{key}' in [{section}]")
            if isinstance(val, list):
                val = tuple(val)
            values[key] = val
    s = Settings(**values)
    unknown = set(s.enabled_setups) - {"breakout", "pullback", "relative_strength"}
    if unknown or not s.enabled_setups:
        raise ValueError(f"config: enabled_setups must be a non-empty subset of breakout/pullback/relative_strength, got {s.enabled_setups}")
    if s.breakeven_trigger_atr < 0:
        raise ValueError("config: breakeven_trigger_atr must be >= 0")
    if s.regime_sma < 0:
        raise ValueError("config: regime_sma must be >= 0")
    return s
