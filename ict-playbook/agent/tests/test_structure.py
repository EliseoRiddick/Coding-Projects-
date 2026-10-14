import numpy as np
import pandas as pd
import pytest

from ict_agent import structure as st

TZ = "America/New_York"


def mk(rows):
    """rows of (open, high, low, close) -> 15m DataFrame."""
    idx = pd.date_range("2025-03-05 09:00", periods=len(rows), freq="15min", tz=TZ)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


MSS_ROWS = [(10, 11, 9.5, 10.5), (10.5, 12, 10, 11.5), (11.5, 11.8, 10.5, 10.8), (10.8, 11.0, 9.0, 9.2),
            (9.2, 10.5, 9.1, 10.3), (10.3, 10.9, 10.0, 10.2), (10.2, 10.4, 9.5, 9.6), (9.6, 9.8, 8.5, 8.7),
            (8.7, 9.5, 8.6, 9.3), (9.3, 9.9, 9.2, 9.8), (9.8, 11.5, 9.7, 11.4)]


def test_pivots_and_nesting():
    df = mk(MSS_ROWS)
    hi, lo = st.pivots(df)
    assert hi == [1, 5] and lo == [3, 7]
    lv = st.swing_levels(df)
    assert lv["short"] == ([1, 5], [3, 7]) and lv["intermediate"] == ([], [])  # 2 pivots cannot be flanked


def test_mss_after_bearish_run():
    events, state = st.structure_events(mk(MSS_ROWS))
    assert [(e["kind"], e["direction"]) for e in events] == [("initial", "bearish"), ("MSS", "bullish")]
    assert events[-1]["level"] == 10.9 and events[-1]["body_close_beyond"] and state == "bullish"


def test_bos_is_continuation():
    rows = MSS_ROWS[:10] + [(9.8, 9.9, 8.0, 8.2)]  # another break lower instead of a shift up
    events, _ = st.structure_events(mk(rows))
    assert events[-1]["kind"] == "BOS" and events[-1]["direction"] == "bearish"


OB_ROWS = [(10, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.0), (10.0, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.0),
           (10.0, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.2), (10.0, 10.1, 9.7, 9.8),  # 6: down-close candle
           (9.8, 11.2, 9.7, 11.0),                                                      # 7: displacement
           (11.0, 11.3, 10.9, 11.2), (11.2, 11.4, 11.0, 11.3)]


def test_order_block_is_last_down_candle_before_displacement():
    df = mk(OB_ROWS)
    disp = st.displacement(df)
    assert list(np.nonzero(disp)[0]) == [7]
    obs = st.order_blocks(df, disp)
    assert len(obs) == 1
    ob = obs[0]
    assert ob["direction"] == "bullish" and ob["top"] == 10.0 and ob["bottom"] == 9.8 and ob["mean_threshold"] == pytest.approx(9.9)
    assert ob["status"] == "untouched" and ob["displacement_clears_recent_extreme"]
    assert ob["time"] == df.index[6].isoformat()


def test_order_block_mitigated_and_broken_by_mean_threshold():
    touched = mk(OB_ROWS + [(11.3, 11.3, 9.95, 10.4)])    # wick back into the block, body holds above MT
    assert st.order_blocks(touched, st.displacement(touched))[0]["status"] == "mitigated"
    broken = mk(OB_ROWS + [(11.3, 11.3, 9.4, 9.5)])        # body closes below MT 9.9 -> R35: no longer valid
    assert [b for b in st.order_blocks(broken, st.displacement(broken)) if b["direction"] == "bullish"] == []


BREAKER_ROWS = [(10.0, 10.4, 9.8, 10.2), (10.2, 12.0, 10.1, 11.8), (11.8, 11.9, 10.5, 10.6), (10.6, 10.7, 9.0, 10.2),
                (10.2, 10.3, 9.1, 9.9), (9.9, 11.0, 9.8, 10.9), (10.9, 12.8, 10.8, 12.7), (12.7, 12.9, 11.9, 12.0),
                (12.0, 12.1, 11.0, 11.2), (11.2, 11.3, 9.9, 10.0), (10.0, 10.2, 8.8, 8.9)]


def breaker_inputs(rows):
    df = mk(rows)
    events, _ = st.structure_events(df, 1, st.displacement(df))
    hi, lo = st.pivots(df)
    return df, events, hi, lo


def test_bearish_breaker_needs_sweep_and_break():
    df, events, hi, lo = breaker_inputs(BREAKER_ROWS)
    assert [(e["kind"], e["direction"]) for e in events] == [("initial", "bullish"), ("MSS", "bearish")]
    b = st.breakers(df, events, hi, lo)
    assert len(b) == 1
    z = b[0]
    assert z["trade_side"] == "sell_zone" and (z["top"], z["bottom"]) == (12.7, 12.0)  # largest down-close body in the run up
    assert z["swept_level"] == 12.0 and z["broken_swing_level"] == 9.0 and z["status"] == "untouched"
    assert z["alternative_zone_lowest_close"] == {"top": 10.2, "bottom": 9.9}  # bar 4 has the lowest close among down candles


def test_breaker_without_sweep_is_ignored():
    rows = [r if i != 6 else (10.9, 11.8, 10.8, 11.7) for i, r in enumerate(BREAKER_ROWS)]
    rows[7] = (11.7, 11.9, 11.2, 11.3)
    rows[8] = (11.3, 11.6, 11.0, 11.2)  # the run up never exceeds the old high (12.0)
    df, events, hi, lo = breaker_inputs(rows)
    assert st.breakers(df, events, hi, lo) == []


def test_breaker_dropped_after_body_close_above_zone():
    df, events, hi, lo = breaker_inputs(BREAKER_ROWS + [(8.9, 13.0, 8.8, 12.9)])
    assert st.breakers(df, events, hi, lo) == []


def test_unicorn_requires_same_direction_gap_overlap():
    df, events, hi, lo = breaker_inputs(BREAKER_ROWS)
    b = st.breakers(df, events, hi, lo)
    st.mark_unicorns(b, [{"type": "bearish", "low": 12.1, "high": 12.5}])
    assert b[0]["unicorn"] is True
    st.mark_unicorns(b, [{"type": "bullish", "low": 12.1, "high": 12.5}, {"type": "bearish", "low": 14, "high": 15}])
    assert b[0]["unicorn"] is False


def test_sweep_closed_back_inside_with_mss_is_turtle_soup():
    df, events, hi, lo = breaker_inputs(BREAKER_ROWS)
    since = df.index[0]
    sw = st.detect_sweeps(df, [{"name": "pdh", "price": 12.0, "side": "buy_side", "since": since},
                               {"name": "never", "price": 20.0, "side": "buy_side", "since": since},
                               {"name": "low_never", "price": 5.0, "side": "sell_side", "since": since}], events)
    assert [s["pool"] for s in sw] == ["pdh"]
    s = sw[0]
    assert s["bar_time"] == df.index[6].isoformat() and s["extreme"] == 12.8
    assert s["closed_back_inside"] and s["opposite_mss_after"] and s["turtle_soup_confirmed"]


def test_sweep_that_keeps_going_is_not_turtle_soup():
    rows = BREAKER_ROWS[:7] + [(12.7, 13.5, 12.6, 13.4), (13.4, 14.0, 13.3, 13.9)]  # closes beyond the pool, no MSS
    df, events, _, _ = breaker_inputs(rows)
    s = st.detect_sweeps(df, [{"name": "pdh", "price": 12.0, "side": "buy_side", "since": df.index[0]}], events)[0]
    assert not s["closed_back_inside"] and not s["turtle_soup_confirmed"]


def test_sweep_ignores_bars_before_the_pool_existed():
    df, events, _, _ = breaker_inputs(BREAKER_ROWS)
    assert st.detect_sweeps(df, [{"name": "late", "price": 12.0, "side": "buy_side", "since": df.index[9]}], events) == []


def test_fvg_levels_and_ce_rule():
    from ict_agent import levels
    rows = [(10, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.0), (10.0, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.0),
            (10.0, 10.2, 9.8, 10.1), (10.1, 10.3, 9.9, 10.2),
            (10.2, 10.4, 10.0, 10.1),                       # candle 1: high 10.4
            (10.1, 11.9, 10.1, 11.8),                       # candle 2: big up, body ~ range
            (11.8, 12.2, 11.4, 12.0)]                       # candle 3: low 11.4 > 10.4 -> gap 10.4-11.4
    g = levels.fair_value_gaps(mk(rows))
    assert len(g) == 1 and g[0]["type"] == "bullish" and g[0]["after_displacement"]
    assert g[0]["levels"] == pytest.approx({"low": 10.4, "q25": 10.65, "ce": 10.9, "q75": 11.15, "high": 11.4})
    ok = levels.fair_value_gaps(mk(rows + [(12.0, 12.1, 11.2, 11.6)]))        # wick into the gap, body above CE: still valid
    assert len(ok) == 1
    dead = levels.fair_value_gaps(mk(rows + [(12.0, 12.1, 10.7, 10.8)]))      # body closes below CE (R35): invalid
    assert dead == []
    doji = rows[:7] + [(10.9, 11.9, 10.1, 11.0), rows[8]]                     # middle candle body is tiny vs its range (R33)
    assert levels.fair_value_gaps(mk(doji)) == []
