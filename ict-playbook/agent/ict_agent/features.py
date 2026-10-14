"""Assemble the full feature payload sent to the model."""
import pandas as pd

from . import levels, sessions
from .data import tf_minutes, truncate


def build_features(frames: dict[str, pd.DataFrame], symbol: str, asof: pd.Timestamp | None = None) -> dict:
    base_name = min(frames, key=tf_minutes)
    if asof is None:
        asof = frames[base_name].index[-1]
    frames = truncate(frames, asof)
    base = frames[base_name]
    if base.empty:
        raise ValueError("No data at or before asof")
    price = float(base["close"].iloc[-1])
    ref = levels.reference_atr(base)
    lv = levels.period_levels(base, asof)
    pb_levels, pb_detail = levels.playbook_levels(base, asof)
    lv.update(pb_levels)
    return {
        "symbol": symbol,
        "current_price": price,
        "base_timeframe": base_name,
        "daily_atr14": ref,
        "time": sessions.time_context(asof),
        "week_profile": sessions.week_profile(base, asof),
        "previous_day": sessions.day_extremes(base, base[base.index < asof.normalize()].index[-1])
        if (base.index < asof.normalize()).any() else None,
        "playbook_detail": pb_detail,
        "key_levels_by_distance": levels.distances(price, lv, ref),
        "timeframes": {k: levels.timeframe_features(v) for k, v in frames.items() if len(v) > 5},
    }
