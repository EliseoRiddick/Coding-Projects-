"""Automatic candle download (Yahoo Finance via yfinance) for index futures.

Yahoo is free but UNOFFICIAL: data is delayed, the service can change or block requests, and continuous
futures contracts show roll gaps. Good for study and replays - use your broker's CSVs (data.load_timeframes)
when you need exact, current data. The history window per interval is Yahoo's: 5m/15m about 60 days, 1h about 2 years."""
from dataclasses import dataclass, field

import pandas as pd

from .data import resample_ohlc, tf_minutes

YAHOO_TICKERS = {"NQ": "NQ=F", "MNQ": "NQ=F", "ES": "ES=F", "MES": "ES=F", "YM": "YM=F", "MYM": "YM=F",
                 "RTY": "RTY=F", "M2K": "RTY=F", "GC": "GC=F", "MGC": "GC=F", "CL": "CL=F"}
# account-currency value of a 1.0 price move for ONE contract (micros are a tenth of the full contract)
POINT_VALUES = {"NQ": 20.0, "MNQ": 2.0, "ES": 50.0, "MES": 5.0, "YM": 5.0, "MYM": 0.5,
                "RTY": 50.0, "M2K": 5.0, "GC": 100.0, "MGC": 10.0, "CL": 1000.0}
PULLS = [("5m", "60d"), ("15m", "60d"), ("1h", "730d"), ("1d", "2y")]  # (interval, Yahoo period)
DERIVED = {"4h": "1h"}  # built locally; Yahoo has no 4h bars
STALE_MINUTES = 30      # while the market is open, warn if the newest bar ended longer ago than this


@dataclass
class MarketData:
    frames: dict
    ticker: str
    last_bar_end: pd.Timestamp
    age_minutes: float
    warnings: list[str] = field(default_factory=list)


def yahoo_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    return YAHOO_TICKERS.get(s, symbol.strip())


def point_value(symbol: str) -> float | None:
    return POINT_VALUES.get(symbol.strip().upper())


def futures_market_open(ts: pd.Timestamp) -> bool:
    """CME equity-index schedule in New York time: Sunday 18:00 to Friday 17:00, with a 17:00-18:00 break."""
    ts = ts.tz_convert("America/New_York")
    wd, hour = ts.weekday(), ts.hour
    if wd == 5 or (wd == 4 and hour >= 17) or (wd == 6 and hour < 18):
        return False
    return hour != 17


def normalise(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Flatten yfinance's (field, ticker) column pairs, lowercase, convert to tz, dedupe and sort."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Yahoo data is missing columns {missing}")
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(tz)
    df.index.name = "time"
    return df[~df.index.duplicated()].sort_index()


def drop_forming_bar(df: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Remove the newest bar if it has not finished yet - an unfinished candle would make every level wrong."""
    if df.empty:
        return df
    end = df.index[-1] + pd.Timedelta(minutes=tf_minutes(interval))
    return df.iloc[:-1] if end > now else df


def _yahoo_history(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # late import so the rest of the package works without it

    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, actions=False)


def fetch_yahoo(symbol: str, tz: str = "America/New_York", now: pd.Timestamp | None = None,
                history=_yahoo_history) -> MarketData:
    """Download 5m/15m/1h/1d candles (4h is derived from 1h). `history` is injectable for tests."""
    now = (now or pd.Timestamp.now(tz=tz)).tz_convert(tz)
    ticker = yahoo_ticker(symbol)
    frames, warnings = {}, []
    for interval, period in PULLS:
        try:
            df = drop_forming_bar(normalise(history(ticker, interval, period), tz), interval, now)
        except Exception as e:  # network, rate limit, blocked, schema change
            warnings.append(f"Yahoo {interval} download failed ({type(e).__name__}: {str(e)[:120]})")
            continue
        if len(df) < 20:
            warnings.append(f"Yahoo returned too little {interval} data for {ticker} ({len(df)} bars)")
            continue
        frames[interval] = df
    if not frames:
        raise ValueError(f"No data from Yahoo for '{symbol}' ({ticker}). It may be blocked or rate-limited, or the "
                         "ticker is wrong. Put CSV exports in your Drive data folder instead. Details: " + "; ".join(warnings))
    for name, src in DERIVED.items():
        if src in frames and name not in frames:
            built = resample_ohlc(frames[src], "4h")
            built = built[built.index >= frames[src].index[0]]  # the first bucket is partial when history starts mid-bucket
            frames[name] = drop_forming_bar(built, name, now)   # ...and so may the newest one
    frames = dict(sorted(frames.items(), key=lambda kv: tf_minutes(kv[0])))

    base = min(frames, key=tf_minutes)
    last_end = frames[base].index[-1] + pd.Timedelta(minutes=tf_minutes(base))
    age = (now - last_end).total_seconds() / 60
    if futures_market_open(now) and age > STALE_MINUTES:
        warnings.append(f"The newest {base} candle ended {age:.0f} minutes ago while the market is open - Yahoo data is "
                        "delayed. Do not treat the entry levels as live prices.")
    elif not futures_market_open(now):
        warnings.append("Market is closed; analysing the last completed session.")
    warnings.append("Continuous futures contract: roll gaps at contract changes can look like real gaps.")
    if len(frames) < 4:
        warnings.append(f"Only {', '.join(frames)} data was available; some timeframes are missing.")
    return MarketData(frames, ticker, last_end, age, warnings)
