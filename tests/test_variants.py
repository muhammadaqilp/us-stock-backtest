"""Rule-variant switches (regime_sma, enabled_setups). SYNTHETIC fixtures only."""
import numpy as np
import pandas as pd
import pytest

from swing_screener.config import load_settings
from swing_screener.indicators import add_indicators
from swing_screener.plan import checklist, setups_from_checklist
from swing_screener.screener import rank_candidates
from swing_screener.strategies import passes_common_filters, qualify_all

from .synthetic import S, random_walk

LOOSE = dict(min_avg_dollar_volume=0, min_price=1.0, breakout_volume_multiple=1.0, general_relative_volume=0.5)


def test_regime_off_by_default_adds_no_column():
    s = S()
    assert s.regime_sma == 0
    ind = add_indicators(random_walk(300, 1), s, random_walk(300, 2, price=300))
    assert "bench_regime_ok" not in ind


def test_regime_flag_is_causal_and_matches_spy_vs_its_sma():
    s = S(regime_sma=50)
    spy = random_walk(400, 5, price=300)
    ind = add_indicators(random_walk(400, 4), s, spy)
    sma = spy["Close"].rolling(50).mean()
    expect = (spy["Close"] > sma).where(sma.notna())
    pd.testing.assert_series_equal(ind["bench_regime_ok"], expect.reindex(ind.index), check_names=False)
    assert ind["bench_regime_ok"].iloc[:49].isna().all()
    short = add_indicators(random_walk(400, 4).iloc[:300], s, spy.iloc[:300])
    pd.testing.assert_series_equal(short["bench_regime_ok"], ind["bench_regime_ok"].iloc[:300])


def test_regime_blocks_entries_when_spy_below_sma_and_unknown_never_passes():
    s = S(regime_sma=200, **LOOSE)
    r = add_indicators(random_walk(400, 4), s, random_walk(400, 5, price=300)).iloc[-1].copy()
    r["bench_regime_ok"] = False
    assert passes_common_filters(r, s) == (False, "market_regime")
    r["bench_regime_ok"] = np.nan
    assert passes_common_filters(r, s) == (False, "market_regime")
    assert qualify_all(r, s) == []


def test_regime_filter_on_falling_market_produces_no_candidates():
    s_off, s_on = S(**LOOSE), S(regime_sma=50, **LOOSE)
    falling_spy = random_walk(500, 9, drift=-0.003, vol=0.004, price=300)
    stocks = {f"T{i}": random_walk(500, 20 + i, drift=0.002) for i in range(8)}
    on = {t: add_indicators(f, s_on, falling_spy) for t, f in stocks.items()}
    diag = {}
    assert rank_candidates(on, s_on, diagnostics=diag).empty
    assert diag.get("filter:market_regime", 0) > 0
    off = {t: add_indicators(f, s_off, falling_spy) for t, f in stocks.items()}
    assert "bench_regime_ok" not in next(iter(off.values()))


def test_disabled_setup_never_qualifies_anywhere():
    s_all = S(**LOOSE)
    s_two = S(enabled_setups=("breakout", "pullback"), **LOOSE)
    seen_rs = 0
    for seed in range(4):
        ind = add_indicators(random_walk(600, seed, drift=0.001, vol=0.02), s_all,
                             random_walk(600, 100 + seed, drift=0.0004, vol=0.008, price=300))
        for i in range(260, 600, 3):
            r = ind.iloc[i]
            seen_rs += "relative_strength" in qualify_all(r, s_all)
            assert qualify_all(r, s_two) == [k for k in qualify_all(r, s_all) if k != "relative_strength"]
        cands = rank_candidates({"X": ind}, s_two)
        assert "relative_strength" not in set(cands.get("setup", []))
    assert seen_rs > 0                     # the switch actually removed something


@pytest.mark.parametrize("kw", [dict(regime_sma=50), dict(enabled_setups=("pullback",)),
                                dict(regime_sma=50, enabled_setups=("breakout", "pullback"))])
def test_checklist_agrees_with_qualifiers_with_variants_on(kw):
    s = S(**LOOSE, **kw)
    for seed in range(3):
        ind = add_indicators(random_walk(600, seed, drift=0.001, vol=0.02), s,
                             random_walk(600, 100 + seed, drift=0.0004, vol=0.008, price=300))
        for i in range(260, 600, 3):
            r = ind.iloc[i]
            assert setups_from_checklist(checklist(r, s)) == qualify_all(r, s), (seed, i)


def test_config_rejects_bad_variant_values(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[strategy]\nenabled_setups = ["breakout", "momo"]\n')
    with pytest.raises(ValueError, match="enabled_setups"):
        load_settings(p)
    p.write_text('[strategy]\nenabled_setups = []\n')
    with pytest.raises(ValueError, match="enabled_setups"):
        load_settings(p)
    p.write_text('[strategy]\nbreakeven_trigger_atr = -1\n')
    with pytest.raises(ValueError, match="breakeven_trigger_atr"):
        load_settings(p)
    p.write_text('[strategy]\nregime_sma = -1\n')
    with pytest.raises(ValueError, match="regime_sma"):
        load_settings(p)


def test_shipped_config_keeps_variants_off():
    s = load_settings("config.toml")
    assert s.regime_sma == 0 and set(s.enabled_setups) == {"breakout", "pullback", "relative_strength"}
    assert s.breakeven_trigger_atr == 0
