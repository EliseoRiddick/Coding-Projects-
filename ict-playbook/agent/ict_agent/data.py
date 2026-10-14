"""Load, normalise and resample OHLC CSVs."""
import re
from pathlib import Path

import pandas as pd

TIME_COLS = ("timestamp", "datetime", "time", "date", "gmt time", "local time")
UNIT_MINUTES = {"m": 1, "min": 1, "h": 60, "d": 1440, "w": 10080}
RESAMPLE_RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON"}


def tf_minutes(name: str) -> int:
    m = re.fullmatch(r"(\d+)\s*(min|m|h|d|w)", name.lower())
    if not m:
        raise ValueError(f"Cannot parse timeframe from '{name}' (expected e.g. 15m, 1h, 4h, 1d)")
    return int(m.group(1)) * UNIT_MINUTES[m.group(2)]


def load_csv(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if tcol is None or missing:
        raise ValueError(f"{path}: need a time column {TIME_COLS} and open/high/low/close (missing: {missing})")
    if pd.api.types.is_numeric_dtype(df[tcol]):  # unix epoch (TradingView exports)
        unit = "ms" if df[tcol].iloc[0] > 1e11 else "s"
        ts = pd.to_datetime(df[tcol], unit=unit, utc=True).dt.tz_convert(tz)
    else:
        raw = df[tcol].astype(str)
        if raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").any():  # explicit UTC offsets (may differ across DST)
            ts = pd.to_datetime(raw, utc=True).dt.tz_convert(tz)
        else:
            ts = pd.to_datetime(raw).dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = pd.DatetimeIndex(ts, name="time")
    out = out[out.index.notna()]
    return out[~out.index.duplicated()].sort_index()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame(
        {"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(), "close": r["close"].last()}
    )
    return out.dropna()


def load_timeframes(data_dir: Path, tz: str, derive: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every `<timeframe>.csv` in data_dir; optionally derive missing timeframes from the finest."""
    frames = {p.stem.lower(): load_csv(p, tz) for p in sorted(Path(data_dir).glob("*.csv"))}
    if not frames:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    for name in frames:
        tf_minutes(name)
    base_name = min(frames, key=tf_minutes)
    for name in derive or []:
        if name not in frames and tf_minutes(name) > tf_minutes(base_name):
            frames[name] = resample_ohlc(frames[base_name], RESAMPLE_RULES.get(name, name))
    return dict(sorted(frames.items(), key=lambda kv: tf_minutes(kv[0])))


def truncate(frames: dict[str, pd.DataFrame], asof: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Drop everything after `asof` so replays cannot look ahead."""
    return {k: v[v.index <= asof] for k, v in frames.items()}
