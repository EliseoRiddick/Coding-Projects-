"""Economic calendar from a folder of CSV exports, plus the playbook's news rules applied in code.

    R64  do not trade the 8:30 candle on data days - wait 2-3 minutes after any release
    R65  FOMC: wait at least 15-30 minutes after the 2:00 decision
    R66  NFP week: stop taking new trades from Wednesday about 11:00, never carry a position into the release
    R67  holiday weeks / bank holidays: half size or skip (reported as a flag, not a hold)

Times in the files are read as the `tz` given (New York by default) unless a UTC offset is written in the value.
Judgement calls the playbook leaves open are constants below (ASSUMPTIONS)."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RELEASE_HOLD_MINUTES = 3   # R64: wait 2-3 minutes after a release (ASSUMPTION: applied to every medium/high release)
FOMC_HOLD_MINUTES = 30     # R65: "at least 15-30 minutes" after the decision (ASSUMPTION: the long end)
NFP_HOLD_START = (2, 11, 0)  # R66: Wednesday (weekday 2) 11:00
LOOK_BACK_DAYS, LOOK_AHEAD_DAYS, MAX_EVENTS = 1, 7, 40
US_NAMES = {"USD", "US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}

HIGH_WORDS = ("nonfarm", "non-farm", "payroll", "nfp", "cpi", "consumer price", "fomc", "federal funds", "fed funds",
              "interest rate decision", "pce", "gdp", "ppi", "producer price", "retail sales", "ism ")
MEDIUM_WORDS = ("jobless claims", "adp", "jolts", "durable goods", "housing starts", "consumer confidence", "consumer sentiment",
                "powell", "fed chair", "speaks", "testifies", "minutes", "unemployment rate", "pmi")
NFP_RE = re.compile(r"non[- ]?farm|\bnfp\b|employment situation", re.I)
FOMC_RE = re.compile(r"fomc|federal funds|fed funds|interest rate decision", re.I)
FOMC_NOT_RE = re.compile(r"minutes|speak|member|testif|press conference", re.I)
HOLIDAY_RE = re.compile(r"holiday|markets? closed|bank holiday", re.I)


@dataclass
class Calendar:
    df: pd.DataFrame | None
    files: list[str] = field(default_factory=list)
    bad_rows: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.df is not None and not self.df.empty


# ------------------------------------------------------------------ reading
def _role(norm: str) -> str | None:
    if norm.startswith("datetime") or norm in ("timestamp", "dateandtime"):
        return "datetime"
    if norm.startswith("date"):
        return "date"
    if norm.startswith("time"):
        return "time"
    if norm in ("currency", "ccy", "cur", "country", "region"):
        return "currency"
    if norm.startswith(("impact", "importance")) or norm in ("volatility", "priority", "stars", "imp"):
        return "impact"
    if norm in ("event", "title", "name", "indicator", "release", "description", "eventname"):
        return "event"
    if norm.startswith("actual"):
        return "actual"
    if norm in ("forecast", "consensus", "expected", "estimate", "fcst", "survey"):
        return "forecast"
    if norm.startswith(("previous", "prior")) or norm == "prev":
        return "previous"
    return None


def map_columns(columns) -> dict[str, str]:
    roles: dict[str, str] = {}
    for col in columns:
        r = _role(re.sub(r"[^a-z0-9]", "", str(col).lower()))
        if r and r not in roles:
            roles[r] = col
    return roles


def _clock(text) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d{1,2})(?:[:.](\d{2}))?(?::\d{2})?\s*([ap]\.?m\.?)?\s*", str(text), re.I)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower().replace(".", "")
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return (h, mi) if h < 24 and mi < 60 else None


def _infer_impact(event: str) -> str:
    e = event.lower() + " "
    if any(w in e for w in HIGH_WORDS) and "speak" not in e and "member" not in e:
        return "high"
    return "medium" if any(w in e for w in MEDIUM_WORDS) else "low"


def _impact(raw, event: str) -> str:
    s = str(raw).strip().lower()
    if s in ("", "nan", "none", "n/a", "-"):
        return _infer_impact(event)
    if "holiday" in s or "non-economic" in s or "non economic" in s:
        return "holiday"
    stars = max(s.count("*"), s.count("★"))
    digit = re.fullmatch(r"([0-3])(?:\.0)?(?:\s*stars?)?", s)
    if "high" in s or s == "red" or stars >= 3 or (digit and digit.group(1) == "3"):
        return "high"
    if "med" in s or s == "orange" or stars == 2 or (digit and digit.group(1) == "2"):
        return "medium"
    if "low" in s or s == "yellow" or stars == 1 or digit:
        return "low"
    return _infer_impact(event)


def _timestamp(date_text, clock_text, dt_text, tz: str):
    """-> (Timestamp tz-aware or NaT, all_day, ok)."""
    try:
        if dt_text is not None:
            raw = str(dt_text)
            if re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", raw.strip()):
                return pd.to_datetime(raw, utc=True).tz_convert(tz), False, True
            ts = pd.to_datetime(raw)
            return ts.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward"), False, True
        day = pd.to_datetime(str(date_text))
        if pd.isna(day):
            return pd.NaT, False, False
        clk = _clock(clock_text) if clock_text is not None else None
        if clk is None:  # "All Day", "Tentative", empty
            return day.tz_localize(tz), True, True
        return (day + pd.Timedelta(hours=clk[0], minutes=clk[1])).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward"), False, True
    except (ValueError, TypeError, OverflowError):
        return pd.NaT, False, False


def _read_one(path: Path, tz: str) -> tuple[pd.DataFrame, int]:
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
    else:
        raise ValueError(f"cannot read {path.name}: {last_err}")
    roles = map_columns(raw.columns)
    if "event" not in roles or not ({"datetime", "date"} & roles.keys()):
        raise ValueError(f"{path.name}: need an event/title column and a date (or datetime) column; found columns {list(raw.columns)}")
    rows, bad = [], 0
    get = lambda r, k: (r[roles[k]] if k in roles else "")
    for _, r in raw.iterrows():
        event = str(get(r, "event")).strip()
        ts, all_day, ok = _timestamp(get(r, "date") if "date" in roles else None, get(r, "time") if "time" in roles else None,
                                     get(r, "datetime") if "datetime" in roles else None, tz)
        if not event or not ok or pd.isna(ts):
            bad += 1
            continue
        rows.append({"time": ts, "date": ts.date(), "all_day": all_day, "event": event,
                     "currency": str(get(r, "currency")).strip().upper(), "impact": _impact(get(r, "impact"), event),
                     "actual": str(get(r, "actual")).strip(), "forecast": str(get(r, "forecast")).strip(),
                     "previous": str(get(r, "previous")).strip()})
    return pd.DataFrame(rows), bad


def load_calendar(folder: Path | str, tz: str = "America/New_York") -> Calendar:
    """Read every CSV in `folder`. Never raises: problems become messages so a bad file cannot block a run."""
    folder = Path(folder)
    if not folder.is_dir():
        return Calendar(None, messages=[f"No economic calendar folder at {folder}."])
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        return Calendar(None, messages=[f"No .csv files in {folder} - news rules R64-R66 cannot be checked."])
    frames, files, bad_total, msgs = [], [], 0, []
    for p in paths:
        try:
            df, bad = _read_one(p, tz)
        except Exception as e:  # unrecognised layout, unreadable file
            msgs.append(f"Skipped {p.name}: {e}")
            continue
        if df.empty:
            msgs.append(f"Skipped {p.name}: no usable rows ({bad} unreadable)")
            continue
        frames.append(df)
        files.append(p.name)
        bad_total += bad
    if not frames:
        return Calendar(None, bad_rows=bad_total, messages=msgs + ["No usable economic calendar - news rules R64-R66 cannot be checked."])
    allr = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "currency", "event"]).sort_values("time", kind="stable")
    if bad_total:
        msgs.append(f"{bad_total} calendar row(s) could not be read and were ignored.")
    return Calendar(allr.reset_index(drop=True), files, bad_total, msgs)


# ------------------------------------------------------------------ rules
def _is_us(cur: str) -> bool:
    return cur == "" or cur in US_NAMES


def _iso(ts) -> str:
    return ts.isoformat()


def summarize(cal: Calendar, asof: pd.Timestamp, currencies=US_NAMES) -> dict:
    """What Claude sees under `features["economic"]`, including the code-applied news holds."""
    if not cal.available:
        return {"available": False, "reason": " ".join(cal.messages) or "No economic calendar provided.",
                "note": "News rules R64-R66 could not be checked; say so in the answer."}
    df = cal.df
    asof = asof.tz_convert(df["time"].dt.tz) if asof.tzinfo is not None else asof.tz_localize(df["time"].dt.tz)
    cur = {c.upper() for c in currencies}
    us = df[df["currency"].map(lambda c: c == "" or c in cur)]
    warnings = list(cal.messages)
    if us.empty:
        warnings.append("The calendar has no US events (currency USD) - check the file.")
    today = asof.normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    week_end = week_start + pd.Timedelta(days=5)
    window = us[(us["time"] >= today - pd.Timedelta(days=LOOK_BACK_DAYS)) & (us["time"] < today + pd.Timedelta(days=LOOK_AHEAD_DAYS + 1))]
    keep = window[window["impact"].isin(["medium", "high", "holiday"])]
    if not len(window):
        warnings.append(f"The calendar has no US events between {(today - pd.Timedelta(days=LOOK_BACK_DAYS)).date()} and "
                        f"{(today + pd.Timedelta(days=LOOK_AHEAD_DAYS)).date()} - it may be out of date.")
    if us["time"].max() < today:
        warnings.append(f"The newest calendar event is {us['time'].max().date()}, before today - update the file.")

    timed = keep[~keep["all_day"]]
    holds = []

    def hold(rule, reason, start, end):
        if start <= asof < end:
            holds.append({"rule": rule, "reason": reason, "until": _iso(end)})

    # R66 - NFP week
    wk = us[(us["time"] >= week_start) & (us["time"] < week_end) & us["event"].str.contains(NFP_RE)]
    nfp = None
    if len(wk):
        nfp = wk.iloc[0]["time"]
        start = week_start + pd.Timedelta(days=NFP_HOLD_START[0], hours=NFP_HOLD_START[1], minutes=NFP_HOLD_START[2])
        if nfp > start and not wk.iloc[0]["all_day"]:
            hold("R66", f"NFP week: no new trades from Wednesday 11:00 until the {nfp:%a %H:%M} release is out",
                 start, nfp + pd.Timedelta(minutes=RELEASE_HOLD_MINUTES))
    # R65 - FOMC decision
    fomc = timed[timed["event"].str.contains(FOMC_RE) & ~timed["event"].str.contains(FOMC_NOT_RE) & (timed["date"] == today.date())]
    fomc_time = fomc.iloc[0]["time"] if len(fomc) else None
    if fomc_time is not None:
        hold("R65", f"FOMC decision at {fomc_time:%H:%M}: wait at least {FOMC_HOLD_MINUTES} minutes after it",
             fomc_time, fomc_time + pd.Timedelta(minutes=FOMC_HOLD_MINUTES))
    # R64 - every medium/high release today
    day_events = timed[timed["date"] == today.date()]
    for _, e in day_events.iterrows():
        hold("R64", f"'{e['event']}' released at {e['time']:%H:%M}: do not trade the candle, wait {RELEASE_HOLD_MINUTES} minutes",
             e["time"], e["time"] + pd.Timedelta(minutes=RELEASE_HOLD_MINUTES))
    holds.sort(key=lambda h: ["R66", "R65", "R64"].index(h["rule"]))

    def ev(e) -> dict:
        d = {"time": _iso(e["time"]) if not e["all_day"] else str(e["date"]), "all_day": bool(e["all_day"]),
             "impact": e["impact"], "event": e["event"]}
        if not e["all_day"]:
            d["minutes_from_now"] = int(round((e["time"] - asof).total_seconds() / 60))
        d.update({k: e[k] for k in ("actual", "forecast", "previous") if e[k]})
        return d

    near = keep.assign(_d=(keep["time"] - asof).abs()).nsmallest(MAX_EVENTS, "_d").sort_values("time")
    high = timed[timed["impact"] == "high"]
    nxt = high[high["time"] > asof].head(1)
    prev = high[high["time"] <= asof].tail(1)
    holiday = us[(us["time"] >= week_start) & (us["time"] < week_end) & ((us["impact"] == "holiday") | us["event"].str.contains(HOLIDAY_RE))]
    return {
        "available": True,
        "files": cal.files,
        "events_in_calendar": int(len(df)),
        "flags": {
            "data_day_0830": bool(len(day_events[day_events["time"].dt.strftime("%H:%M") == "08:30"])),
            "fomc_today": fomc_time is not None,
            "nfp_week": nfp is not None,
            "nfp_release": _iso(nfp) if nfp is not None else None,
            "holiday_week": bool(len(holiday)),
            "holiday_names": sorted(set(holiday["event"]))[:3],
            "high_impact_today": [f"{e['time']:%H:%M} {e['event']}" for _, e in day_events[day_events["impact"] == "high"].iterrows()],
        },
        "news_hold": holds[0] if holds else None,
        "next_high_impact": ev(nxt.iloc[0]) if len(nxt) else None,
        "last_high_impact": ev(prev.iloc[0]) if len(prev) else None,
        "events": [ev(e) for _, e in near.iterrows()],
        "warnings": warnings,
    }
