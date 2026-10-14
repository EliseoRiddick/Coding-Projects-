"""Time-of-day / day-of-week features, following playbook Chapter 5 (all clock times New York).

Windows below are the playbook's "working choice" values (section 5.2). Edit them here if you revise
the playbook. Rule numbers (R52, R53, ...) are cited so the model and you can trace each flag."""
import pandas as pd

M = lambda h, m=0: h * 60 + m  # minute of day

# Non-overlapping partition used to label where a high/low formed (5.2).
SESSIONS = {
    "asia": (M(20), M(24)),
    "london": (M(2), M(5)),
    "ny_premarket": (M(7), M(9)),
    "ny_morning": (M(9, 30), M(11, 30)),
    "lunch": (M(11, 30), M(13, 30)),
    "ny_afternoon": (M(13, 30), M(16)),
}

# Overlapping named windows reported as flags (5.2, 5.4, 5.6, 5.7).
WINDOWS = {
    "london_open": (M(2), M(5)),
    "ny_open_premarket": (M(7), M(9)),
    "opening_range": (M(9, 30), M(10)),
    "index_morning": (M(9, 30), M(11)),
    "london_close": (M(10), M(12)),
    "lunch_no_new_trades": (M(11, 30), M(13, 30)),
    "afternoon": (M(13, 30), M(15)),
    "final_hour": (M(15), M(16)),
    "final_hour_sweet_spot": (M(15, 15), M(15, 45)),
    "silver_bullet_london": (M(3), M(4)),
    "silver_bullet_am": (M(10), M(11)),
    "silver_bullet_pm": (M(14), M(15)),
}
# R52: tradable windows; R53: no new trades 11:30-13:30
TRADE_WINDOWS = ["index_morning", "afternoon", "final_hour_sweet_spot"]
MACRO_HOURS = range(7, 16)  # 5.5: h:50 to (h+1):10 for h in 7..15

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_GUIDANCE = {  # R62, 5.9
    "Monday": "Small-range manipulation day; let it trade without you unless you are advanced (R62).",
    "Tuesday": "Strongest day with Wednesday (R62).",
    "Wednesday": "Strongest day with Tuesday (R62).",
    "Thursday": "Trade only if range remains and the week's objective is unmet (R62).",
    "Friday": "Small unless the target is unmet; TGIF retrace 20-30% (40% max) of the week's range, only after a strong one-directional week (R62, R68).",
    "Saturday": "Market closed.",
    "Sunday": "No trading; conditional scenarios only (5.9).",
}


def _in(m: int, win: tuple[int, int]) -> bool:
    a, b = win
    return a <= m < b if a < b else (m >= a or m < b)


def session_of(ts: pd.Timestamp) -> str:
    m = ts.hour * 60 + ts.minute
    return next((n for n, w in SESSIONS.items() if _in(m, w)), "off_session")


def active_windows(ts: pd.Timestamp) -> list[str]:
    m = ts.hour * 60 + ts.minute
    return [n for n, w in WINDOWS.items() if _in(m, w)]


def in_macro(ts: pd.Timestamp) -> bool:
    return (ts.hour in MACRO_HOURS and ts.minute >= 50) or (ts.hour - 1 in MACRO_HOURS and ts.minute <= 10)


def trading_status(ts: pd.Timestamp) -> dict:
    """What R52/R53/R64 say about trading right now."""
    m = ts.hour * 60 + ts.minute
    act = active_windows(ts)
    if ts.weekday() >= 5:
        status = "closed_or_conditional_only"
    elif "lunch_no_new_trades" in act:
        status = "no_new_trades"  # R53
    elif any(w in act for w in TRADE_WINDOWS):
        status = "tradable_window"  # R52
    elif "ny_open_premarket" in act:
        status = "observe_only"  # R52: 7:00-9:00 is observation
    else:
        status = "outside_windows"
    return {
        "status": status,
        "rules": ["R52", "R53"],
        "data_day_830_caution": M(8, 30) <= m < M(8, 33),  # R64: do not trade the 8:30 candle on data days
        "in_macro_window": in_macro(ts),  # R58: timing aid only
    }


def calendar_caution(ts: pd.Timestamp) -> str | None:
    """5.12 / R67 [LOW CONFIDENCE]. Holiday dates and Jackson Hole week are not known here - supply them as economic data."""
    if ts.month in (7, 8):
        return "July-August is treated as light-trade (R67)."
    if (ts.month == 12 and ts.day >= 20) or ts.month == 1 or (ts.month == 2 and ts.day <= 14):
        return "Late December to mid-February lull (5.12, low confidence)."
    return None


def next_window(ts: pd.Timestamp) -> dict | None:
    m = ts.hour * 60 + ts.minute
    ups = sorted(((a - m) % 1440, n) for n, (a, _) in WINDOWS.items() if n not in active_windows(ts))
    return {"window": ups[0][1], "minutes_until": ups[0][0]} if ups else None


def time_context(asof: pd.Timestamp) -> dict:
    return {
        "asof": asof.isoformat(),
        "day_of_week": DAYS[asof.weekday()],
        "week_day_number": asof.weekday() + 1,
        "day_guidance": DAY_GUIDANCE[DAYS[asof.weekday()]],
        "clock": asof.strftime("%H:%M"),
        "current_session": session_of(asof),
        "active_windows": active_windows(asof),
        "trading_window": trading_status(asof),
        "next_window": next_window(asof),
        "calendar_caution": calendar_caution(asof),
        "window_definitions_minutes_of_day": {k: list(v) for k, v in WINDOWS.items()},
    }


def day_extremes(base: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    """High/low of a calendar day and the session in which each formed."""
    d = base[base.index.normalize() == day.normalize()]
    if d.empty:
        return None
    hi, lo = d["high"].idxmax(), d["low"].idxmin()
    return {
        "date": str(day.date()),
        "weekday": DAYS[day.weekday()],
        "open": float(d["open"].iloc[0]),
        "high": float(d["high"].max()),
        "high_time": hi.strftime("%H:%M"),
        "high_session": session_of(hi),
        "low": float(d["low"].min()),
        "low_time": lo.strftime("%H:%M"),
        "low_session": session_of(lo),
        "close": float(d["close"].iloc[-1]),
    }


def week_profile(base: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Days of the current week so far, plus where the weekly high/low sit (5.8, 5.9)."""
    week_start = (asof - pd.Timedelta(days=asof.weekday())).normalize()
    days = []
    for i in range(asof.weekday() + 1):
        e = day_extremes(base[base.index <= asof], week_start + pd.Timedelta(days=i))
        if e:
            days.append(e)
    wk = base[(base.index >= week_start) & (base.index <= asof)]
    out = {"days_so_far": days}
    if not wk.empty:
        out.update(
            week_high=float(wk["high"].max()),
            week_high_weekday=DAYS[wk["high"].idxmax().weekday()],
            week_low=float(wk["low"].min()),
            week_low_weekday=DAYS[wk["low"].idxmin().weekday()],
        )
    return out
