import numpy as np
import pandas as pd
import pytest

from ict_agent import levels, sessions
from ict_agent.data import load_csv, resample_ohlc, tf_minutes
from ict_agent.features import build_features
from ict_agent.prompt import parse_json_response

TZ = "America/New_York"


def synth(start="2025-03-03 00:00", days=14, freq="15min", seed=1):
    idx = pd.date_range(start, periods=days * 96, freq=freq, tz=TZ)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.random(len(idx)) * 0.2
    low = np.minimum(open_, close) - rng.random(len(idx)) * 0.2
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx.rename("time"))


def test_tf_minutes():
    assert tf_minutes("15m") == 15 and tf_minutes("4h") == 240 and tf_minutes("1d") == 1440
    with pytest.raises(ValueError):
        tf_minutes("weird")


def test_resample_preserves_extremes():
    df = synth()
    h = resample_ohlc(df, "1h")
    assert h["high"].max() == df["high"].max() and h["low"].min() == df["low"].min()
    assert len(h) == len(df) // 4


def test_load_csv_naive_and_epoch(tmp_path):
    df = synth(days=1)
    p = tmp_path / "15m.csv"
    df.reset_index().assign(time=lambda x: x["time"].dt.tz_localize(None)).to_csv(p, index=False)
    got = load_csv(p, TZ)
    assert got.index[0].hour == 0 and str(got.index.tz) == TZ
    e = tmp_path / "e.csv"
    pd.DataFrame({"time": [int(t.timestamp()) for t in df.index], **{c: df[c].values for c in df}}).to_csv(e, index=False)
    assert load_csv(e, TZ).index[0] == df.index[0]


@pytest.mark.parametrize("hhmm,name", [("03:00", "london_killzone"), ("08:30", "ny_am_killzone"),
                                       ("21:00", "asia"), ("14:00", "ny_pm_killzone"), ("17:00", "off_session")])
def test_session_of(hhmm, name):
    assert sessions.session_of(pd.Timestamp(f"2025-03-04 {hhmm}", tz=TZ)) == name


def test_session_across_dst():
    # US DST starts 2025-03-09; wall-clock sessions must not shift
    assert sessions.session_of(pd.Timestamp("2025-03-10 08:30", tz=TZ)) == "ny_am_killzone"


def test_fvg_detection_and_fill():
    idx = pd.date_range("2025-03-03", periods=6, freq="15min", tz=TZ)
    df = pd.DataFrame({"open": [10, 10.5, 12, 12.5, 12.6, 12.7], "high": [10.5, 11, 13, 13, 13, 13],
                       "low": [9.5, 10.4, 11.5, 12.2, 12.3, 12.4], "close": [10.4, 12, 12.9, 12.6, 12.7, 12.8]}, index=idx)
    gaps = levels.fair_value_gaps(df)
    assert any(g["type"] == "bullish" and g["low"] == 10.5 and g["high"] == 11.5 for g in gaps)
    df.loc[idx[-1], "low"] = 10.0  # price trades back through the gap
    assert not any(g["low"] == 10.5 for g in levels.fair_value_gaps(df))


def test_period_levels_prev_day():
    df = synth()
    asof = pd.Timestamp("2025-03-12 10:00", tz=TZ)  # Wednesday
    L = levels.period_levels(df[df.index <= asof], asof)
    prev = df[df.index.normalize() == pd.Timestamp("2025-03-11", tz=TZ)]
    assert L["prev_day_high"] == prev["high"].max() and L["prev_day_low"] == prev["low"].min()
    assert "prev_week_high" in L and "midnight_open" in L and "0830_open" in L


def test_build_features_no_lookahead():
    f15 = synth()
    frames = {"15m": f15, "1h": resample_ohlc(f15, "1h")}
    asof = pd.Timestamp("2025-03-07 09:00", tz=TZ)
    feats = build_features(frames, "TEST", asof)
    assert feats["current_price"] == f15.loc[asof, "close"]
    assert feats["time"]["day_of_week"] == "Friday" and feats["time"]["current_session"] == "ny_am_killzone"
    assert set(feats["timeframes"]) == {"15m", "1h"}
    d = [abs(r["distance"]) for r in feats["key_levels_by_distance"]]
    assert d == sorted(d) and d


def test_parse_json_response():
    assert parse_json_response('Sure:\n{"a": 1}\n') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_response("no json")


def test_load_csv_offsets_across_dst(tmp_path):
    df = synth(start="2025-03-08 12:00", days=3, freq="1h")
    p = tmp_path / "1h.csv"
    df.reset_index().to_csv(p, index=False)  # strings carry -05:00 then -04:00
    got = load_csv(p, TZ)
    assert len(got) == len(df) and got.index[0] == df.index[0] and got.index[-1] == df.index[-1]
