"""Deterministic market-structure detection, following the playbook:
swing nesting (R11), MSS (R12-R13), liquidity sweeps / turtle soup (R26), order blocks (R35, R39),
breakers (R41) and the unicorn flag (R42).

Everything the playbook leaves unspecified is a constant below - an ASSUMPTION you can edit."""
import numpy as np
import pandas as pd

DISPLACEMENT_ATR = 1.5  # ASSUMPTION: "displacement" = candle body >= 1.5 x the prior 14-bar ATR (playbook gives no number)
PIVOT_N = 1             # R11: a short-term high has a lower high on each side (1 bar each side)
OB_LOOKBACK = 5         # how far back from a displacement leg to look for the order-block candle
LEG_CONTEXT = 20        # bars used to test "displacement clears a recent high/low" (R33)
BREAKER_LOOKBACK = 60   # how far back before the swing the "old high/low" may sit (R41)


def _sign(x):
    return np.sign(x).astype(int)


def atr_prev(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """14-bar average true range excluding the current bar (so a big candle is not compared with itself)."""
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=3).mean().shift(1)


def displacement(df: pd.DataFrame, k: float = DISPLACEMENT_ATR) -> np.ndarray:
    """+1 / -1 for a bullish / bearish displacement candle, else 0."""
    body = (df["close"] - df["open"]).to_numpy()
    thr = (k * atr_prev(df)).to_numpy()
    with np.errstate(invalid="ignore"):
        hit = np.abs(body) >= thr  # NaN thresholds compare False
    return np.where(hit, _sign(body), 0)


# ---------------------------------------------------------------- swings (R11)
def pivots(df: pd.DataFrame, n: int = PIVOT_N) -> tuple[list[int], list[int]]:
    """Positions of short-term highs / lows: strictly beyond the n bars on each side."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    hi, lo = [], []
    for i in range(n, len(df) - n):
        if all(h[i] > h[i - k] and h[i] > h[i + k] for k in range(1, n + 1)):
            hi.append(i)
        if all(l[i] < l[i - k] and l[i] < l[i + k] for k in range(1, n + 1)):
            lo.append(i)
    return hi, lo


def _nest(prices: np.ndarray, pos: list[int], high: bool) -> list[int]:
    """Promote a pivot one level when both neighbouring pivots of the same level are lower (higher for lows)."""
    out = []
    for j in range(1, len(pos) - 1):
        p, a, b = prices[pos[j]], prices[pos[j - 1]], prices[pos[j + 1]]
        if (p > a and p > b) if high else (p < a and p < b):
            out.append(pos[j])
    return out


def swing_levels(df: pd.DataFrame, n: int = PIVOT_N) -> dict[str, tuple[list[int], list[int]]]:
    hi, lo = pivots(df, n)
    H, L = df["high"].to_numpy(), df["low"].to_numpy()
    ih, il = _nest(H, hi, True), _nest(L, lo, False)
    return {"short": (hi, lo), "intermediate": (ih, il), "long": (_nest(H, ih, True), _nest(L, il, False))}


def _pt(df, pos, col):
    return {"time": df.index[pos].isoformat(), "price": float(df[col].iloc[pos])}


def swing_summary(df: pd.DataFrame, levels: dict, last: int = 3) -> dict:
    return {name: {"highs": [_pt(df, p, "high") for p in hi[-last:]], "lows": [_pt(df, p, "low") for p in lo[-last:]]}
            for name, (hi, lo) in levels.items()}


# ---------------------------------------------------------------- MSS / BOS (R12-R13)
def structure_events(df: pd.DataFrame, n: int = PIVOT_N, disp: np.ndarray | None = None):
    """Walk the bars, tracking the most recent confirmed short-term high/low. A break of the high is a bullish
    shift (MSS) if the prior break was bearish, otherwise a continuation (BOS). Per R12 a wick break counts;
    `body_close_beyond` records whether a candle body also closed beyond the level so you can weigh it."""
    h, l, o, c = (df[k].to_numpy() for k in ("high", "low", "open", "close"))
    hi, lo = pivots(df, n)
    confirm_h, confirm_l = {p + n: p for p in hi}, {p + n: p for p in lo}
    ref_h = ref_l = state = None
    events = []
    for i in range(len(df)):
        ref_h = confirm_h.get(i, ref_h)
        ref_l = confirm_l.get(i, ref_l)
        up = ref_h is not None and h[i] > h[ref_h]
        dn = ref_l is not None and l[i] < l[ref_l]
        if up and dn:  # outside bar: take the side the candle closed toward
            up, dn = c[i] >= o[i], c[i] < o[i]
        if up:
            kind = "initial" if state is None else ("MSS" if state == "bearish" else "BOS")
            events.append({"kind": kind, "direction": "bullish", "pos": i, "time": df.index[i], "level": float(h[ref_h]),
                           "swing_pos": ref_h, "body_close_beyond": bool(c[i] > h[ref_h]),
                           "displacement": bool(disp is not None and disp[i] > 0)})
            state, ref_h = "bullish", None
        elif dn:
            kind = "initial" if state is None else ("MSS" if state == "bullish" else "BOS")
            events.append({"kind": kind, "direction": "bearish", "pos": i, "time": df.index[i], "level": float(l[ref_l]),
                           "swing_pos": ref_l, "body_close_beyond": bool(c[i] < l[ref_l]),
                           "displacement": bool(disp is not None and disp[i] < 0)})
            state, ref_l = "bearish", None
    return events, state


def _public_event(e: dict) -> dict:
    return {"kind": e["kind"], "direction": e["direction"], "time": e["time"].isoformat(), "broken_level": e["level"],
            "body_close_beyond": e["body_close_beyond"], "displacement": e["displacement"]}


# ---------------------------------------------------------------- order blocks (R35, R39)
def order_blocks(df: pd.DataFrame, disp: np.ndarray, last: int = 3) -> list[dict]:
    """Per R39: the last opposite-close candle (consecutive ones merged) before a displacement leg.
    Zone = candle open to the block's far body edge; mean threshold (MT) = 50% of that body.
    A body close beyond the MT ends validity (R35), so broken blocks are dropped."""
    o, c, h, l = (df[k].to_numpy() for k in ("open", "close", "high", "low"))
    n, out, i = len(df), [], 0
    while i < n:
        d = int(disp[i])
        if d == 0:
            i += 1
            continue
        s = i
        while s > 0 and np.sign(c[s - 1] - o[s - 1]) == d:
            s -= 1
        e = i
        while e + 1 < n and np.sign(c[e + 1] - o[e + 1]) == d:
            e += 1
        j = s - 1
        while j >= max(0, s - OB_LOOKBACK) and np.sign(c[j] - o[j]) != -d:
            j -= 1
        if j >= max(0, s - OB_LOOKBACK) and j >= 0:
            k = j
            while k - 1 >= 0 and np.sign(c[k - 1] - o[k - 1]) == -d:
                k -= 1
            if d == 1:
                top, bottom = float(o[k]), float(c[k:j + 1].min())
            else:
                bottom, top = float(o[k]), float(c[k:j + 1].max())
            mt = (top + bottom) / 2
            later_c, later_h, later_l = c[e + 1:], h[e + 1:], l[e + 1:]
            broken = bool((later_c < mt).any() if d == 1 else (later_c > mt).any())
            touched = bool((later_l <= top).any() if d == 1 else (later_h >= bottom).any())
            ctx = slice(max(0, s - LEG_CONTEXT), s)
            clears = bool(h[s:e + 1].max() > h[ctx].max()) if d == 1 and s > 0 else \
                bool(l[s:e + 1].min() < l[ctx].min()) if s > 0 else False
            if not broken:
                out.append({"direction": "bullish" if d == 1 else "bearish", "time": df.index[k].isoformat(),
                            "top": top, "bottom": bottom, "mean_threshold": mt,
                            "status": "mitigated" if touched else "untouched", "displacement_clears_recent_extreme": clears,
                            "leg_start": df.index[s].isoformat()})
        i = e + 1
    res = []
    for direction in ("bullish", "bearish"):
        res += [b for b in out if b["direction"] == direction][-last:]
    return sorted(res, key=lambda b: b["time"])


# ---------------------------------------------------------------- breakers (R41, R42)
def breakers(df: pd.DataFrame, events: list[dict], hi_pos: list[int], lo_pos: list[int], last: int = 3) -> list[dict]:
    """R41, bearish case: an old high A, a swing low B, a higher high that sweeps A, then a break of B (MSS).
    The down-close candle of that run up (largest body; lowest close given as the alternative) is the sell zone.
    Bullish mirrors. A body close beyond the zone afterwards ends it."""
    o, c, h, l = (df[k].to_numpy() for k in ("open", "close", "high", "low"))
    out = []
    for ev in events:
        if ev["kind"] != "MSS":
            continue
        i, sp, bear = ev["pos"], ev["swing_pos"], ev["direction"] == "bearish"
        if i - sp < 2:
            continue
        if bear:
            old = [p for p in hi_pos if sp - BREAKER_LOOKBACK <= p < sp]
            if not old:
                continue
            a = old[-1]
            peak = sp + int(np.argmax(h[sp:i]))
            if h[peak] <= h[a]:
                continue
            cands = [m for m in range(sp, peak + 1) if c[m] < o[m]]
            swept = float(h[a])
        else:
            old = [p for p in lo_pos if sp - BREAKER_LOOKBACK <= p < sp]
            if not old:
                continue
            a = old[-1]
            peak = sp + int(np.argmin(l[sp:i]))
            if l[peak] >= l[a]:
                continue
            cands = [m for m in range(sp, peak + 1) if c[m] > o[m]]
            swept = float(l[a])
        if not cands:
            continue
        pick = max(cands, key=lambda m: abs(c[m] - o[m]))
        alt = (min(cands, key=lambda m: c[m]) if bear else max(cands, key=lambda m: c[m]))
        top, bottom = (o[pick], c[pick]) if bear else (c[pick], o[pick])
        top, bottom = float(max(top, bottom)), float(min(top, bottom))
        later_c = c[i + 1:]
        broken = bool((later_c > top).any() if bear else (later_c < bottom).any())
        if broken:
            continue
        touched = bool((h[i + 1:] >= bottom).any() if bear else (l[i + 1:] <= top).any())
        item = {"direction": ev["direction"], "trade_side": "sell_zone" if bear else "buy_zone",
                "time": df.index[pick].isoformat(), "top": top, "bottom": bottom, "mean_threshold": (top + bottom) / 2,
                "swept_level": swept, "swept_level_time": df.index[a].isoformat(),
                "broken_swing_level": ev["level"], "mss_time": ev["time"].isoformat(),
                "status": "mitigated" if touched else "untouched", "selection": "largest_body"}
        if alt != pick:
            at, ab = (max(o[alt], c[alt]), min(o[alt], c[alt]))
            item["alternative_zone_lowest_close" if bear else "alternative_zone_highest_close"] = {"top": float(at), "bottom": float(ab)}
        out.append(item)
    return out[-last:]


def mark_unicorns(breaker_list: list[dict], fvgs: list[dict]) -> None:
    """R42: a breaker overlapping a gap of the same direction at the same location."""
    for b in breaker_list:
        b["unicorn"] = any(g["type"] == b["direction"] and g["low"] <= b["top"] and g["high"] >= b["bottom"] for g in fvgs)


# ---------------------------------------------------------------- liquidity sweeps (R26)
def pivot_pools(df: pd.DataFrame, levels: dict, tol: float, n: int = PIVOT_N) -> list[dict]:
    """Pools from this timeframe's own swings: intermediate swings, plus short-term swings within `tol` of
    each other (equal highs/lows, R25, using the outer edge). `since` = when the pool became known."""
    pools, ih_il = [], levels["intermediate"]
    for side, pos, col in (("buy_side", ih_il[0][-4:], "high"), ("sell_side", ih_il[1][-4:], "low")):
        for p in pos:
            pools.append({"name": f"swing_{col}", "price": float(df[col].iloc[p]), "side": side,
                          "since": df.index[min(p + n, len(df) - 1)]})
    sh, sl = levels["short"]
    for side, pos, col in (("buy_side", sh[-6:], "high"), ("sell_side", sl[-6:], "low")):
        for a, b in zip(pos, pos[1:]):
            pa, pb = float(df[col].iloc[a]), float(df[col].iloc[b])
            if tol and abs(pa - pb) <= tol:
                pools.append({"name": f"equal_{col}s", "price": max(pa, pb) if col == "high" else min(pa, pb), "side": side,
                              "since": df.index[min(b + n, len(df) - 1)]})
    return pools


def detect_sweeps(df: pd.DataFrame, pools: list[dict], events: list[dict], max_out: int = 8) -> list[dict]:
    """A sweep = a bar trades beyond a pool after the pool formed. R26: it is a turtle-soup setup only if price
    closes back inside (within 2 bars) AND an opposite-direction MSS follows."""
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    n, out = len(df), []
    for p in pools:
        buy = p["side"] == "buy_side"
        start = int(df.index.searchsorted(p["since"]))
        if start >= n:
            continue
        beyond = (h[start:] > p["price"]) if buy else (l[start:] < p["price"])
        if not beyond.any():
            continue
        pos = start + int(np.argmax(beyond))
        win = range(pos, min(pos + 3, n))
        back = any((c[m] <= p["price"]) if buy else (c[m] >= p["price"]) for m in win)
        opposite = "bearish" if buy else "bullish"
        mss = any(e["kind"] == "MSS" and e["direction"] == opposite and e["pos"] >= pos for e in events)
        out.append({"pool": p["name"], "pool_price": p["price"], "side": p["side"], "bar_time": df.index[pos].isoformat(),
                    "extreme": float(h[pos] if buy else l[pos]), "closed_back_inside": bool(back),
                    "opposite_mss_after": bool(mss), "turtle_soup_confirmed": bool(back and mss)})
    out.sort(key=lambda s: s["bar_time"], reverse=True)
    return out[:max_out]


# ---------------------------------------------------------------- one call per timeframe
def analyze(df: pd.DataFrame, fvgs: list[dict], tol: float, named_pools: list[dict] | None = None) -> dict:
    disp = displacement(df)
    lv = swing_levels(df)
    events, state = structure_events(df, PIVOT_N, disp)
    brk = breakers(df, events, lv["short"][0], lv["short"][1])
    mark_unicorns(brk, fvgs)
    pools = pivot_pools(df, lv, tol) + (named_pools or [])
    return {
        "swings_by_level": swing_summary(df, lv),
        "structure": {"bias": state, "recent_events": [_public_event(e) for e in events[-4:]]},
        "order_blocks": order_blocks(df, disp),
        "breakers": brk,
        "sweeps": detect_sweeps(df, pools, events),
        "recent_displacement_candles": [{"time": df.index[i].isoformat(), "direction": "bullish" if disp[i] > 0 else "bearish"}
                                        for i in np.nonzero(disp)[0][-3:]],
    }
