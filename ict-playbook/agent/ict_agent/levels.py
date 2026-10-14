"""Deterministic price-level features."""
import numpy as np
import pandas as pd

from .data import resample_ohlc


def atr(df: pd.DataFrame, n: int = 14) -> float | None:
    if len(df) < 2:
        return None
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=2).mean().iloc[-1]
    return None if pd.isna(v) else float(v)


def _between(df, a, b):
    return df[(df.index >= a) & (df.index < b)]


def _hl(df, label):
    return {f"{label}_high": float(df["high"].max()), f"{label}_low": float(df["low"].min())} if not df.empty else {}


def period_levels(base: pd.DataFrame, asof: pd.Timestamp) -> dict[str, float]:
    """Named reference prices from the finest-timeframe data (previous day/week/month, opens)."""
    L: dict[str, float] = {}
    today = asof.normalize()
    prior_days = base[base.index < today]
    if not prior_days.empty:
        d = prior_days.index[-1].normalize()
        L.update(_hl(_between(base, d, d + pd.Timedelta(days=1)), "prev_day"))
    week_start = (today - pd.Timedelta(days=today.weekday())).normalize()
    L.update(_hl(_between(base, week_start - pd.Timedelta(days=7), week_start), "prev_week"))
    month_start = today.replace(day=1)
    prev_month_start = (month_start - pd.Timedelta(days=1)).replace(day=1)
    L.update(_hl(_between(base, prev_month_start, month_start), "prev_month"))
    for label, t in (("midnight_open", today), ("0830_open", today + pd.Timedelta(hours=8, minutes=30)),
                     ("0930_open", today + pd.Timedelta(hours=9, minutes=30))):
        if t <= asof:
            bars = base[(base.index >= t) & (base.index <= asof)]
            if not bars.empty:
                L[label] = float(bars["open"].iloc[0])
    wk = base[(base.index >= week_start) & (base.index <= asof)]
    if not wk.empty:
        L["week_open"] = float(wk["open"].iloc[0])
        L["week_high_so_far"], L["week_low_so_far"] = float(wk["high"].max()), float(wk["low"].min())
    td = base[(base.index >= today) & (base.index <= asof)]
    if not td.empty:
        L["day_high_so_far"], L["day_low_so_far"] = float(td["high"].max()), float(td["low"].min())
    return L


def swings(df: pd.DataFrame, n: int = 2, last: int = 4) -> dict[str, list[dict]]:
    """Fractal swing highs/lows confirmed by n bars on each side."""
    w = 2 * n + 1
    hi = df["high"][df["high"] == df["high"].rolling(w, center=True).max()]
    lo = df["low"][df["low"] == df["low"].rolling(w, center=True).min()]
    fmt = lambda s: [{"time": t.isoformat(), "price": float(p)} for t, p in s.tail(last).items()]
    return {"swing_highs": fmt(hi), "swing_lows": fmt(lo)}


def equal_levels(sw: list[dict], tol: float) -> list[float]:
    """Prices where 2+ recent swings sit within tol of each other (resting liquidity)."""
    prices = sorted(s["price"] for s in sw)
    return [round((a + b) / 2, 5) for a, b in zip(prices, prices[1:]) if tol and b - a <= tol]


def fair_value_gaps(df: pd.DataFrame, lookback: int = 150, last: int = 3) -> list[dict]:
    """Unfilled 3-candle imbalances. Filled = later price traded through the whole gap."""
    d = df.tail(lookback)
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    out = []
    for i in range(2, len(d)):
        if l[i] > h[i - 2]:  # bullish
            lo_, hi_, kind = h[i - 2], l[i], "bullish"
            filled = i + 1 < len(d) and l[i + 1:].min() <= lo_
        elif h[i] < l[i - 2]:  # bearish
            lo_, hi_, kind = h[i], l[i - 2], "bearish"
            filled = i + 1 < len(d) and h[i + 1:].max() >= hi_
        else:
            continue
        if not filled:
            out.append({"type": kind, "low": float(lo_), "high": float(hi_), "time": d.index[i - 1].isoformat()})
    return out[-last:]


def range_position(df: pd.DataFrame, bars: int = 50) -> dict:
    d = df.tail(bars)
    lo, hi = float(d["low"].min()), float(d["high"].max())
    c = float(d["close"].iloc[-1])
    pct = (c - lo) / (hi - lo) if hi > lo else 0.5
    return {"range_low": lo, "range_high": hi, "equilibrium": (lo + hi) / 2, "close_pct_of_range": round(pct, 3),
            "zone": "premium" if pct > 0.5 else "discount"}


def reference_atr(base: pd.DataFrame) -> float | None:
    """Daily ATR, used to express distances in comparable units."""
    return atr(resample_ohlc(base, "1D"))


def distances(price: float, levels: dict[str, float], ref_atr: float | None) -> list[dict]:
    rows = []
    for name, p in levels.items():
        d = p - price
        rows.append({"level": name, "price": round(p, 5), "distance": round(d, 5),
                     "distance_daily_atr": round(d / ref_atr, 3) if ref_atr else None,
                     "relative": "above" if d > 0 else "below" if d < 0 else "at"})
    return sorted(rows, key=lambda r: abs(r["distance"]))


def timeframe_features(df: pd.DataFrame, bars_shown: int = 5) -> dict:
    a = atr(df)
    sw = swings(df)
    tol = 0.1 * a if a else 0.0
    return {
        "last_close": float(df["close"].iloc[-1]),
        "atr14": a,
        "recent_bars": [{"time": t.isoformat(), **{k: float(v) for k, v in r.items()}}
                        for t, r in df.tail(bars_shown).iterrows()],
        **sw,
        "equal_highs": equal_levels(sw["swing_highs"], tol),
        "equal_lows": equal_levels(sw["swing_lows"], tol),
        "unfilled_fair_value_gaps": fair_value_gaps(df),
        "dealing_range": range_position(df),
    }
