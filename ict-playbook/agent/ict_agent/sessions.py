"""Time-of-day / day-of-week features. All clock times are in the data's timezone
(America/New_York by default). The session windows are common conventions; the
playbook is authoritative where it differs - adjust SESSIONS to match it."""
import pandas as pd

# name -> (start_minute_of_day, end_minute_of_day); end may be < start for overnight windows
SESSIONS = {
    "asia": (20 * 60, 24 * 60),
    "london_killzone": (2 * 60, 5 * 60),
    "ny_am_killzone": (7 * 60, 10 * 60),
    "ny_lunch_london_close": (10 * 60, 12 * 60),
    "ny_pm_killzone": (13 * 60 + 30, 16 * 60),
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def session_of(ts: pd.Timestamp) -> str:
    m = ts.hour * 60 + ts.minute
    for name, (a, b) in SESSIONS.items():
        if (a <= m < b) if a < b else (m >= a or m < b):
            return name
    return "off_session"


def time_context(asof: pd.Timestamp) -> dict:
    nxt = None
    m = asof.hour * 60 + asof.minute
    upcoming = sorted(((a - m) % 1440, n) for n, (a, _) in SESSIONS.items() if session_of(asof) != n)
    if upcoming:
        nxt = {"session": upcoming[0][1], "minutes_until": upcoming[0][0]}
    return {
        "asof": asof.isoformat(),
        "day_of_week": DAYS[asof.weekday()],
        "week_day_number": asof.weekday() + 1,
        "clock": asof.strftime("%H:%M"),
        "current_session": session_of(asof),
        "next_session": nxt,
        "session_definitions_minutes_of_day": {k: list(v) for k, v in SESSIONS.items()},
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
    """Days of the current week so far, plus where the weekly high/low sit."""
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
