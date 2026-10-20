"""Ticker universe: current Nasdaq Trader directories -> filtered snapshot. NOT point-in-time (survivorship bias)."""
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

NASDAQ_TRADED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
# Security-name patterns that are not plain common equity.
NON_COMMON = (r"warrant|\bright|\bunit\b|units\b|preferred|\bnotes?\b|depositary|trust|fund|etf|etn|"
              r"bond|debenture|acquisition corp|beneficial interest|limited partnership|lp$")


def _read_pipe(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "us-swing-screener/0.2 research"})
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)


def filter_listings(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Pure function (unit-tested): combine the two directory frames into ticker/exchange/name rows."""
    parts = []
    if {"Symbol", "ETF", "Test Issue", "Security Name"}.issubset(a.columns):
        x = a[(a["Test Issue"] == "N") & (a["ETF"] == "N")]
        if "Financial Status" in x.columns:
            x = x[x["Financial Status"].fillna("N") == "N"]  # exclude deficient/delinquent/bankrupt flags
        parts.append(pd.DataFrame({"ticker": x["Symbol"], "exchange": "NASDAQ/other", "name": x["Security Name"]}))
    if {"ACT Symbol", "ETF", "Test Issue", "Security Name"}.issubset(b.columns):
        x = b[(b["Test Issue"] == "N") & (b["ETF"] == "N")]
        parts.append(pd.DataFrame({"ticker": x["ACT Symbol"], "exchange": x.get("Exchange", "OTHER"), "name": x["Security Name"]}))
    if not parts:
        raise ValueError("Could not parse Nasdaq Trader symbol directories (unexpected columns)")
    out = pd.concat(parts, ignore_index=True).dropna(subset=["ticker"])
    out["ticker"] = out["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    out = out[out["ticker"].str.fullmatch(r"[A-Z]{1,5}(-[A-Z])?")]
    out = out[~out["name"].fillna("").str.lower().str.contains(NON_COMMON, regex=True)]
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def build_universe(output: str | Path = "data/universe.csv") -> pd.DataFrame:
    out = filter_listings(_read_pipe(NASDAQ_TRADED), _read_pipe(OTHER_LISTED))
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    p.with_suffix(".meta.json").write_text(json.dumps({
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": [NASDAQ_TRADED, OTHER_LISTED], "rows": len(out),
        "note": "current listings only; not point-in-time; delisted names absent"}, indent=2))
    return out


def load_universe(path: str | Path = "data/universe.csv") -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `swing-screener universe --refresh` first.")
    df = pd.read_csv(p, dtype=str)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must contain a ticker column")
    return sorted(set(df["ticker"].dropna().str.strip()) - {""})


def sample_universe(tickers: list[str], n: int | None, seed: int) -> list[str]:
    """Deterministic seeded sample (alphabetical truncation would bias toward 'A' tickers)."""
    tickers = sorted(tickers)
    if not n or n >= len(tickers):
        return tickers
    return sorted(pd.Series(tickers).sample(n=n, random_state=seed).tolist())


def snapshot(tickers, used, failed, path: Path) -> dict:
    """Persist the exact ticker set requested/used in a run, with a content hash."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": sorted(tickers), "used": [t in set(used) for t in sorted(tickers)]}).to_csv(path / "universe_snapshot.csv", index=False)
    info = {"requested": len(tickers), "used": len(used), "failed_or_excluded": len(failed),
            "sha256_16": hashlib.sha256("\n".join(sorted(tickers)).encode()).hexdigest()[:16],
            "membership": "CURRENT listings (survivorship-biased); not point-in-time"}
    (path / "universe_snapshot.json").write_text(json.dumps(info, indent=2))
    return info


def load_names(path: str | Path = "data/universe.csv") -> dict:
    """ticker -> company/security name from the universe snapshot ({} if unavailable)."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    if not {"ticker", "name"}.issubset(df.columns):
        return {}
    return dict(zip(df["ticker"], df["name"].fillna("").str.strip()))
