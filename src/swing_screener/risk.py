"""Fills, costs, stop/target levels and whole-share position sizing (shared by scan and backtest)."""
import math


def buy_fill(price: float, rate: float) -> float:
    return price * (1 + rate)


def sell_fill(price: float, rate: float) -> float:
    return price * (1 - rate)


def levels(entry: float, atr: float, s) -> dict:
    """Stop/target from an entry price and ATR. Gross reward/risk is the pre-cost ratio of the multiples."""
    stop = entry - s.atr_stop_multiple * atr
    target = entry + s.atr_target_multiple * atr
    if not (atr > 0 and stop > 0):
        return {"valid": False}
    r = s.cost_rate
    gross = (target - entry) / (entry - stop)
    net = (sell_fill(target, r) - buy_fill(entry, r)) / (buy_fill(entry, r) - sell_fill(stop, r))
    return {"valid": True, "stop": stop, "target": target, "rr_gross": round(gross, 6), "rr_net": round(net, 4)}


def size_position(equity: float, cash: float, entry_raw: float, stop_raw: float, s) -> dict:
    """Whole shares limited by (a) risk budget = risk_per_trade * equity, using risk/share INCLUDING entry and
    stop-exit costs, (b) max_position_fraction * equity, (c) available cash. Never relaxes limits to reach 1 share."""
    r = s.cost_rate
    bf = buy_fill(entry_raw, r)
    rps = bf - sell_fill(stop_raw, r)
    if equity <= 0 or bf <= 0 or rps <= 0:
        return {"shares": 0, "reason": "invalid_inputs", "buy_fill": bf}
    by_risk = math.floor(equity * s.risk_per_trade / rps)
    by_cap = math.floor(equity * s.max_position_fraction / bf)
    by_cash = math.floor(max(cash, 0.0) / bf)
    shares = max(0, min(by_risk, by_cap, by_cash))
    reason = ""
    if shares == 0:
        reason = min((("risk_budget", by_risk), ("position_cap", by_cap), ("cash", by_cash)), key=lambda x: x[1])[0]
    return {"shares": shares, "reason": reason, "buy_fill": bf, "risk_per_share": rps,
            "by_risk": by_risk, "by_cap": by_cap, "by_cash": by_cash}
