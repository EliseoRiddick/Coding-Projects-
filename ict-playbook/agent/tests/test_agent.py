import json
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


@pytest.mark.parametrize("hhmm,name", [("03:00", "london"), ("08:30", "ny_premarket"), ("09:45", "ny_morning"),
                                       ("21:00", "asia"), ("12:00", "lunch"), ("14:00", "ny_afternoon"),
                                       ("17:00", "off_session")])
def test_session_of(hhmm, name):
    assert sessions.session_of(pd.Timestamp(f"2025-03-04 {hhmm}", tz=TZ)) == name


def test_session_across_dst():
    # US DST starts 2025-03-09; wall-clock sessions must not shift
    assert sessions.session_of(pd.Timestamp("2025-03-10 08:30", tz=TZ)) == "ny_premarket"


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
    asof = pd.Timestamp("2025-03-07 08:45", tz=TZ)
    feats = build_features(frames, "TEST", asof)
    assert feats["current_price"] == pytest.approx(f15.loc[asof, "close"], abs=1e-4)
    assert feats["time"]["day_of_week"] == "Friday" and feats["time"]["current_session"] == "ny_premarket"
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


def test_infer_timeframe_names():
    from ict_agent.data import infer_tf_name
    assert infer_tf_name(synth(freq="15min")) == "15m"
    assert infer_tf_name(resample_ohlc(synth(), "1h")) == "1h"
    assert infer_tf_name(resample_ohlc(synth(), "4h")) == "4h"
    assert infer_tf_name(resample_ohlc(synth(), "1D")) == "1d"


def test_load_timeframes_labels_oddly_named_files(tmp_path):
    from ict_agent.data import load_timeframes
    f15 = synth()
    f15.to_csv(tmp_path / "NQ1!_export.csv")
    resample_ohlc(f15, "1h").to_csv(tmp_path / "hourly chart.csv")
    frames = load_timeframes(tmp_path, TZ, derive=["4h"])
    assert list(frames) == ["15m", "1h", "4h"]


def test_image_blocks_downscale_and_limit():
    import io
    from PIL import Image
    from ict_agent.images import image_blocks
    buf = io.BytesIO()
    Image.new("RGBA", (3000, 1500), (10, 20, 30, 255)).save(buf, format="PNG")
    blk = image_blocks([buf.getvalue()])[0]
    assert blk["type"] == "image" and blk["source"]["media_type"] == "image/png"
    import base64
    out = Image.open(io.BytesIO(base64.b64decode(blk["source"]["data"])))
    assert max(out.size) == 1568
    with pytest.raises(ValueError):
        image_blocks([buf.getvalue()] * 9)


def test_user_content_order_and_econ():
    from ict_agent.prompt import user_content
    img = {"type": "image", "source": {}}
    c = user_content({"x": 1}, econ="CPI 08:30 high impact", images=[img, img])
    assert c[0]["type"] == c[1]["type"] == "image" and c[2]["type"] == "text"
    assert "<economic_data>" in c[2]["text"] and "2 chart screenshot" in c[2]["text"]
    assert "economic_data" not in user_content({"x": 1})[0]["text"]


def ts(s):
    return pd.Timestamp(s, tz=TZ)


@pytest.mark.parametrize("when,status", [("2025-03-05 10:15", "tradable_window"), ("2025-03-05 12:00", "no_new_trades"),
                                         ("2025-03-05 08:00", "observe_only"), ("2025-03-05 16:30", "outside_windows"),
                                         ("2025-03-05 14:00", "tradable_window"), ("2025-03-08 10:15", "closed_or_conditional_only")])
def test_trading_status_follows_r52_r53(when, status):
    assert sessions.trading_status(ts(when))["status"] == status


def test_macro_silver_bullet_and_830_flags():
    assert sessions.in_macro(ts("2025-03-05 09:55")) and sessions.in_macro(ts("2025-03-05 10:08"))
    assert not sessions.in_macro(ts("2025-03-05 10:30")) and not sessions.in_macro(ts("2025-03-05 04:55"))
    assert "silver_bullet_am" in sessions.active_windows(ts("2025-03-05 10:30"))
    assert sessions.trading_status(ts("2025-03-05 08:31"))["data_day_830_caution"]
    assert not sessions.trading_status(ts("2025-03-05 08:40"))["data_day_830_caution"]
    assert sessions.calendar_caution(ts("2025-08-12")) and sessions.calendar_caution(ts("2025-03-05")) is None


def futures_like(weeks=14, seed=3):
    """15m bars with a 17:00-18:00 daily break and a weekend gap (Sun 18:00 open), with price jumps at reopens."""
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    t = pd.Timestamp("2025-01-05 18:00", tz=TZ)  # a Sunday
    end = t + pd.Timedelta(weeks=weeks)
    while t < end:
        if t.weekday() == 4 and t.hour >= 17 or t.weekday() == 5 or (t.weekday() == 6 and t.hour < 18):
            t += pd.Timedelta(minutes=15); continue
        if t.hour == 17:
            t += pd.Timedelta(minutes=15); continue
        if t.hour == 18 and t.minute == 0:
            price += rng.normal(0, 2.0)  # reopen gap
        o = price; price += rng.normal(0, 0.3)
        rows.append((t, o, max(o, price) + 0.1, min(o, price) - 0.1, price))
        t += pd.Timedelta(minutes=15)
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"]).set_index("time")


def test_playbook_levels_on_futures_like_data():
    df = futures_like()
    asof = ts("2025-04-02 10:30")  # Wednesday, after the opening range completes
    L, detail = levels.playbook_levels(df[df.index <= asof], asof)
    today = df[(df.index >= ts("2025-04-02 07:00")) & (df.index < ts("2025-04-02 09:00"))]
    assert L["premarket_high_0700_0900"] == today["high"].max() and L["premarket_low_0700_0900"] == today["low"].min()
    orr = df[(df.index >= ts("2025-04-02 09:30")) & (df.index < ts("2025-04-02 10:00"))]
    assert L["opening_range_high_0930_1000"] == orr["high"].max() and detail["opening_range_complete"]
    # RTH gap: last bar starting before 16:15 on the prior day -> 09:30 open
    prev = df[(df.index >= ts("2025-04-01 00:00")) & (df.index < ts("2025-04-01 16:15"))]["close"].iloc[-1]
    op = df.loc[ts("2025-04-02 09:30"), "open"]
    if abs(op - prev) >= levels.MIN_GAP:
        assert L["rth_gap_ce"] == pytest.approx((prev + op) / 2)
        assert detail["rth_opening_gap"]["type"] == ("premium" if op > prev else "discount")
    assert detail["ndog_recent"] and detail["nwog_recent"]
    assert all(r["low"] <= r["ce"] <= r["high"] for r in detail["ndog_recent"] + detail["nwog_recent"])
    for n in (20, 40, 60):
        assert L[f"lookback_{n}d_high"] >= L[f"lookback_{n}d_low"]
    assert L["lookback_60d_high"] >= L["lookback_20d_high"] and L["lookback_60d_low"] <= L["lookback_20d_low"]


def test_features_include_playbook_context():
    df = futures_like()
    feats = build_features({"15m": df}, "FUT", ts("2025-04-02 10:30"))
    t = feats["time"]
    assert t["trading_window"]["status"] == "tradable_window" and "silver_bullet_am" in t["active_windows"]
    assert t["day_of_week"] == "Wednesday" and "R62" in t["day_guidance"]
    assert any(r["level"] == "lookback_20d_high" for r in feats["key_levels_by_distance"])
    assert "playbook_detail" in feats


def test_workspace_header_is_sent_only_when_configured(monkeypatch):
    import types
    from unittest import mock
    from ict_agent.analyze import run_analysis

    class S:
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def get_final_message(s):
            return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text="{\"ideas\": []}")],
                                         usage=types.SimpleNamespace(cache_read_input_tokens=0, input_tokens=1, output_tokens=1))

    f15 = synth()
    frames = {"15m": f15}
    asof = pd.Timestamp("2025-03-07 10:30", tz=TZ)
    for env, expected in (("wrkspc_123", {"anthropic-workspace-id": "wrkspc_123"}), ("  ", None), (None, None)):
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        if env is not None:
            monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", env)
        client = mock.MagicMock()
        client.messages.stream.return_value = S()
        with mock.patch("anthropic.Anthropic", return_value=client) as ctor:
            run_analysis(frames, "T", "R1", asof=asof)
        assert ctor.call_args.kwargs.get("default_headers") == expected


def _stub_claude(text='{"market_read": "m", "ideas": [{"bias": "long", "entry_zone": [100, 101], "stop": 99, "targets": [105]}]}'):
    import types
    from unittest import mock

    class S:
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def get_final_message(s):
            return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text=text)],
                                         usage=types.SimpleNamespace(cache_read_input_tokens=0, input_tokens=1, output_tokens=1))

    client = mock.MagicMock()
    client.messages.stream.return_value = S()
    return mock.patch("anthropic.Anthropic", return_value=client)


def test_journal_folder_does_not_break_a_run(tmp_path):
    from ict_agent.analyze import run_analysis
    from ict_agent.risk import RiskConfig
    (tmp_path / "journal.json").mkdir()
    with _stub_claude():
        result, _ = run_analysis({"15m": synth()}, "T", "R1", asof=pd.Timestamp("2025-03-07 10:30", tz=TZ),
                                 out_dir=tmp_path / "out", risk_cfg=RiskConfig(50000, 2), journal_path=tmp_path / "journal.json")
    assert result["ideas"][0]["risk"]["approved"] is True or result["ideas"][0]["risk"]["issues"]   # sized or rejected, never a crash
    assert result["risk_state"]["effective_risk_pct"] == 1.0


def test_guardrail_failure_fails_closed_and_keeps_the_paid_answer(tmp_path):
    from unittest import mock
    from ict_agent.analyze import run_analysis
    from ict_agent.risk import RiskConfig
    with _stub_claude(), mock.patch("ict_agent.risk.apply_guardrails", side_effect=RuntimeError("boom")):
        result, _ = run_analysis({"15m": synth()}, "T", "R1", asof=pd.Timestamp("2025-03-07 10:30", tz=TZ),
                                 out_dir=tmp_path / "out", risk_cfg=RiskConfig(50000, 2))
    r = result["ideas"][0]["risk"]
    assert r["approved"] is False and r["executable_now"] is False and "boom" in r["issues"][0]
    assert result["risk_state"] is None and "boom" in result["risk_error"]
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert len(files) == 2 and files[0].endswith("_raw.json") is False or files[1].endswith("_raw.json")
    raw = next(p for p in (tmp_path / "out").iterdir() if p.name.endswith("_raw.json"))
    assert "risk" not in json.loads(raw.read_text())["ideas"][0]          # raw model answer, saved before the checks
