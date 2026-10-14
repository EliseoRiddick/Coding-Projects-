"""Deterministic price-level features."""
import numpy as np
import pandas as pd

from . import structure
from .data import infer_tf_name, resample_ohlc, tf_minutes

MAX_BARS = 3000  # structure/sweep detection looks at the latest N bars of each timeframe
MIN_GAP = 1.0  # R56: ignore opening gaps under about one handle (index point). Raise/lower for other markets.


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


def equal_levels(pts: list[dict], tol: float) -> list[float]:
    """Prices where 2+ recent swings sit within tol of each other (resting liquidity, R25)."""
    prices = sorted(p["price"] for p in pts)
    return [round((a + b) / 2, 5) for a, b in zip(prices, prices[1:]) if tol and b - a <= tol]


def fair_value_gaps(df: pd.DataFrame, lookback: int = 150, last: int = 3) -> list[dict]:
    """Unfilled 3-candle gaps (R32). Dropped: wick traded through the whole gap, a body closed beyond the CE
    (R35), a doji middle candle (R33), or a gap of only a few ticks (R32). Each carries the R34 levels and an
    `after_displacement` flag (R33) - a gap without it is weaker, not discarded."""
    d = df.tail(lookback)
    o, c, h, l = (d[k].to_numpy() for k in ("open", "close", "high", "low"))
    disp = structure.displacement(d)
    a = structure.atr_prev(d).to_numpy()
    out = []
    for i in range(2, len(d)):
        if l[i] > h[i - 2]:
            lo_, hi_, kind, sgn = h[i - 2], l[i], "bullish", 1
        elif h[i] < l[i - 2]:
            lo_, hi_, kind, sgn = h[i], l[i - 2], "bearish", -1
        else:
            continue
        mid_range = h[i - 1] - l[i - 1]
        if mid_range <= 0 or abs(c[i - 1] - o[i - 1]) < 0.4 * mid_range:  # doji-type middle candle
            continue
        if not np.isnan(a[i - 1]) and (hi_ - lo_) < 0.05 * a[i - 1]:  # only a few ticks
            continue
        ce = (lo_ + hi_) / 2
        later_c, later_h, later_l = c[i + 1:], h[i + 1:], l[i + 1:]
        wick_filled = bool((later_l <= lo_).any() if kind == "bullish" else (later_h >= hi_).any())
        ce_broken = bool((later_c < ce).any() if kind == "bullish" else (later_c > ce).any())
        if wick_filled or ce_broken:
            continue
        span = hi_ - lo_
        out.append({"type": kind, "low": float(lo_), "high": float(hi_), "time": d.index[i - 1].isoformat(),
                    "levels": {"low": float(lo_), "q25": float(lo_ + 0.25 * span), "ce": float(ce),
                               "q75": float(lo_ + 0.75 * span), "high": float(hi_)},
                    "after_displacement": bool(disp[i - 1] == sgn)})
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


def timeframe_features(df: pd.DataFrame, bars_shown: int = 5, named_pools: list[dict] | None = None) -> dict:
    df = df.tail(MAX_BARS)
    a = atr(df)
    tol = 0.1 * a if a else 0.0
    fvgs = fair_value_gaps(df)
    st = structure.analyze(df, fvgs, tol, named_pools)
    short = st["swings_by_level"]["short"]
    return {
        "last_close": float(df["close"].iloc[-1]),
        "atr14": a,
        "recent_bars": [{"time": t.isoformat(), **{k: float(v) for k, v in r.items()}}
                        for t, r in df.tail(bars_shown).iterrows()],
        **st,
        "equal_highs": equal_levels(short["highs"], tol),
        "equal_lows": equal_levels(short["lows"], tol),
        "unfilled_fair_value_gaps": fvgs,
        "dealing_range": range_position(df),
    }


def _window(base, day, start, end, asof):
    """Bars with start <= time-of-day < end on `day`, only up to asof."""
    t0, t1 = day + pd.Timedelta(hours=start[0], minutes=start[1]), day + pd.Timedelta(hours=end[0], minutes=end[1])
    w = base[(base.index >= t0) & (base.index < min(t1, asof + pd.Timedelta(seconds=1)))]
    return w


def playbook_levels(base: pd.DataFrame, asof: pd.Timestamp) -> tuple[dict[str, float], dict]:
    """Chapter 5 reference prices. Returns (name->price for distance ranking, detail dict)."""
    L: dict[str, float] = {}
    detail: dict = {}
    today = asof.normalize()

    pm = _window(base, today, (7, 0), (9, 0), asof)  # R57
    if not pm.empty:
        L["premarket_high_0700_0900"], L["premarket_low_0700_0900"] = float(pm["high"].max()), float(pm["low"].min())
        detail["premarket_range"] = {"complete": asof.hour >= 9, "range": float(pm["high"].max() - pm["low"].min())}
    orr = _window(base, today, (9, 30), (10, 0), asof)  # R54
    if not orr.empty:
        L["opening_range_high_0930_1000"], L["opening_range_low_0930_1000"] = float(orr["high"].max()), float(orr["low"].min())
        detail["opening_range_complete"] = asof.hour >= 10
    mor = _window(base, today, (0, 0), (0, 30), asof)  # 5.4 midnight opening range
    if not mor.empty:
        L["midnight_range_high_0000_0030"], L["midnight_range_low_0000_0030"] = float(mor["high"].max()), float(mor["low"].min())

    # RTH opening range gap (R55): last print at 16:14 on the prior day -> today's 9:30 open
    if asof >= today + pd.Timedelta(hours=9, minutes=30):
        o930 = base[(base.index >= today + pd.Timedelta(hours=9, minutes=30)) & (base.index <= asof)]
        prior = base[base.index < today]
        if not o930.empty and not prior.empty:
            last_day = prior.index[-1].normalize()
            pd_bars = prior[(prior.index >= last_day) & (prior.index < last_day + pd.Timedelta(hours=16, minutes=15))]
            if not pd_bars.empty:
                pc, op = float(pd_bars["close"].iloc[-1]), float(o930["open"].iloc[0])
                if abs(op - pc) >= MIN_GAP:
                    ce = (pc + op) / 2
                    L["rth_gap_prev_close_1614"], L["rth_gap_open_0930"], L["rth_gap_ce"] = pc, op, ce
                    detail["rth_opening_gap"] = {"type": "premium" if op > pc else "discount", "size": round(abs(op - pc), 5),
                                                 "ce": ce, "size_guidance": "20-75 handles tends to return to CE; 75-120 gray; 120+ keeps going (5.4, unverified)"}

    # NDOG / NWOG (R56): gaps across the 17:00-18:00 reopen and the weekend. Needs futures-style data with real gaps.
    recent = base[base.index >= asof - pd.Timedelta(days=45)]  # only the last ~6 weeks of gaps matter (R56)
    tfm = tf_minutes(infer_tf_name(recent)) if len(recent) > 3 else 0
    if tfm:
        end = recent.index + pd.Timedelta(minutes=tfm)
        gap_min = (recent.index[1:] - end[:-1]).total_seconds() / 60
        idx = [int(i) + 1 for i in (gap_min >= 45).nonzero()[0] if recent.index[i + 1] <= asof]
        ndog, nwog = [], []
        for i in idx:
            pc, op = float(recent["close"].iloc[i - 1]), float(recent["open"].iloc[i])
            if abs(op - pc) < MIN_GAP:
                continue
            row = {"date": str(recent.index[i].date()), "low": min(pc, op), "high": max(pc, op), "ce": (pc + op) / 2}
            (nwog if gap_min[i - 1] >= 24 * 60 else ndog).append(row)
        detail["ndog_recent"], detail["nwog_recent"] = ndog[-3:], nwog[-5:]
        for r in ndog[-1:]:
            L["ndog_ce"] = r["ce"]
        for k, r in enumerate(reversed(nwog[-2:])):
            L["nwog_ce" if k == 0 else "nwog_prev_ce"] = r["ce"]

    # 20/40/60 trading-day look-back highs/lows (R63): yesterday = day 1, no Sundays
    daily = resample_ohlc(base[base.index < today], "1D")
    daily = daily[daily.index.weekday != 6]
    for n in (20, 40, 60):
        if len(daily) >= n:
            w = daily.tail(n)
            L[f"lookback_{n}d_high"], L[f"lookback_{n}d_low"] = float(w["high"].max()), float(w["low"].min())
    return L, detail
