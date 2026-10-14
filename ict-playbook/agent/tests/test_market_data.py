import numpy as np
import pandas as pd
import pytest

from ict_agent import market_data as md
from ict_agent.features import build_features

TZ = "America/New_York"


def raw_yahoo(minutes, n, end, ticker="NQ=F", tz_aware=True, seed=0):
    """Shaped like yfinance: (field, ticker) MultiIndex columns, tz-aware (or naive UTC) index, title-case names."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end)
    end = end.tz_localize(TZ) if end.tzinfo is None else end.tz_convert(TZ)
    idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min")
    close = 18000 + np.cumsum(rng.normal(0, 3, n))
    op = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"Open": op, "High": np.maximum(op, close) + 1, "Low": np.minimum(op, close) - 1,
                       "Close": close, "Volume": 100}, index=idx)
    if not tz_aware:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


def fake_history(now, fail=(), short=()):
    mins = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}
    calls = []

    def history(ticker, interval, period):
        calls.append((ticker, interval, period))
        if interval in fail:
            raise ConnectionError("blocked")
        n = 10 if interval in short else {"5m": 600, "15m": 600, "1h": 600, "1d": 300}[interval]
        # newest bar starts at the current bar boundary, so it is still forming at `now`
        step = pd.Timedelta(minutes=mins[interval])
        last_start = now.floor(f"{mins[interval]}min") if interval != "1d" else now.normalize()
        return raw_yahoo(mins[interval], n, last_start)

    history.calls = calls
    return history


def test_ticker_and_point_value_maps():
    assert md.yahoo_ticker("mnq") == "NQ=F" and md.yahoo_ticker("ES") == "ES=F" and md.yahoo_ticker("AAPL") == "AAPL"
    assert md.point_value("NQ") == 20 and md.point_value("mnq") == 2 and md.point_value("XYZ") is None


@pytest.mark.parametrize("when,open_", [("2025-03-08 12:00", False), ("2025-03-09 17:59", False), ("2025-03-09 18:00", True),
                                        ("2025-03-05 17:30", False), ("2025-03-05 10:00", True), ("2025-03-07 16:59", True),
                                        ("2025-03-07 17:00", False)])
def test_futures_schedule(when, open_):
    assert md.futures_market_open(pd.Timestamp(when, tz=TZ)) is open_


def test_normalise_flattens_converts_and_cleans():
    raw = raw_yahoo(15, 40, "2025-03-05 10:00", tz_aware=False)
    raw = pd.concat([raw.iloc[::-1], raw.iloc[:2]])        # unsorted with duplicates
    df = md.normalise(raw, TZ)
    assert list(df.columns) == ["open", "high", "low", "close"] and str(df.index.tz) == TZ
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any() and len(df) == 40
    assert df.index[-1] == pd.Timestamp("2025-03-05 10:00", tz=TZ)
    assert md.normalise(pd.DataFrame(), TZ).empty


def test_normalise_rejects_missing_columns():
    bad = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2025-03-05", periods=2, freq="1h", tz=TZ))
    with pytest.raises(ValueError):
        md.normalise(bad, TZ)


def test_forming_bar_is_dropped_only_while_unfinished():
    df = md.normalise(raw_yahoo(15, 30, "2025-03-05 10:00"), TZ)           # newest candle starts 10:00, ends 10:15
    assert len(md.drop_forming_bar(df, "15m", pd.Timestamp("2025-03-05 10:07", tz=TZ))) == 29
    assert len(md.drop_forming_bar(df, "15m", pd.Timestamp("2025-03-05 10:15", tz=TZ))) == 30


def test_fetch_builds_all_timeframes_and_derives_4h():
    now = pd.Timestamp("2025-03-05 10:07", tz=TZ)
    hist = fake_history(now)
    data = md.fetch_yahoo("NQ", TZ, now=now, history=hist)
    assert list(data.frames) == ["5m", "15m", "1h", "4h", "1d"]
    assert {c[0] for c in hist.calls} == {"NQ=F"} and [c[1] for c in hist.calls] == ["5m", "15m", "1h", "1d"]
    assert all(df.index[-1] + pd.Timedelta(minutes=md.tf_minutes(k)) <= now for k, df in data.frames.items())  # nothing unfinished
    h1, h4 = data.frames["1h"], data.frames["4h"]
    assert h4.index[0] >= h1.index[0]                                   # no partial leading bucket
    start = h4.index[3]
    bucket = h1[(h1.index >= start) & (h1.index < start + pd.Timedelta(hours=4))]
    assert len(bucket) == 4 and h4.loc[start, "high"] == bucket["high"].max() and h4.loc[start, "low"] == bucket["low"].min()
    assert h4.loc[start, "open"] == bucket["open"].iloc[0] and h4.loc[start, "close"] == bucket["close"].iloc[-1]
    assert any("roll gaps" in w for w in data.warnings)
    assert data.last_bar_end == pd.Timestamp("2025-03-05 10:05", tz=TZ) and data.age_minutes == pytest.approx(2.0)
    assert not any("delayed" in w for w in data.warnings)


def test_stale_data_warns_only_when_market_is_open():
    now = pd.Timestamp("2025-03-05 10:07", tz=TZ)
    hist = fake_history(now - pd.Timedelta(hours=2))                   # newest data is 2h old at `now`
    data = md.fetch_yahoo("NQ", TZ, now=now, history=hist)
    assert any("delayed" in w for w in data.warnings)
    closed = pd.Timestamp("2025-03-08 12:00", tz=TZ)                  # Saturday
    d2 = md.fetch_yahoo("NQ", TZ, now=closed, history=fake_history(closed - pd.Timedelta(days=1)))
    assert not any("delayed" in w for w in d2.warnings) and any("closed" in w for w in d2.warnings)


def test_partial_failure_is_a_warning_total_failure_is_an_error():
    now = pd.Timestamp("2025-03-05 10:07", tz=TZ)
    data = md.fetch_yahoo("NQ", TZ, now=now, history=fake_history(now, fail=("5m",)))
    assert "5m" not in data.frames and any("5m download failed" in w for w in data.warnings)
    short = md.fetch_yahoo("NQ", TZ, now=now, history=fake_history(now, short=("1d",)))
    assert "1d" not in short.frames and any("too little 1d" in w for w in short.warnings)
    with pytest.raises(ValueError, match="CSV"):
        md.fetch_yahoo("NQ", TZ, now=now, history=fake_history(now, fail=("5m", "15m", "1h", "1d")))


def test_fetched_frames_feed_the_feature_builder():
    now = pd.Timestamp("2025-03-05 10:07", tz=TZ)
    data = md.fetch_yahoo("NQ", TZ, now=now, history=fake_history(now))
    f = build_features(data.frames, "NQ")
    assert f["base_timeframe"] == "5m" and set(f["timeframes"]) == {"5m", "15m", "1h", "4h", "1d"}
    assert f["time"]["asof"].startswith("2025-03-05T10:00")   # start of the last COMPLETED 5m bar (10:05 is still forming)
