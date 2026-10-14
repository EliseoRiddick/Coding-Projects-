"""Assemble the full feature payload sent to the model."""
import re

import pandas as pd

from . import levels, sessions
from .data import tf_minutes, truncate

# When a named level becomes a pool that can be swept (hours after midnight of the analysed day)
POOL_SINCE_HOURS = {"premarket": 9.0, "opening_range": 10.0, "midnight_range": 0.5}
SWEEP_MAX_TF_MINUTES = 60  # sweep detection against named levels on timeframes up to 1h


def named_pools(lv: dict[str, float], asof: pd.Timestamp) -> list[dict]:
    """Turn highs/lows in the level dict into buy-side / sell-side pools (R21, R24). Running extremes
    ('so_far') are excluded because they are the current price extremes, not resting liquidity."""
    today, pools = asof.normalize(), []
    for name, price in lv.items():
        if "so_far" in name:
            continue
        side = "buy_side" if re.search(r"high", name) else "sell_side" if re.search(r"low", name) else None
        if side is None:
            continue
        hours = next((v for k, v in POOL_SINCE_HOURS.items() if name.startswith(k)), 0.0)
        pools.append({"name": name, "price": price, "side": side, "since": today + pd.Timedelta(hours=hours)})
    return pools


def _round(x, nd=5):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


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
    pools = named_pools(lv, asof)
    tfs = {k: levels.timeframe_features(v, named_pools=pools if tf_minutes(k) <= SWEEP_MAX_TF_MINUTES else None)
           for k, v in frames.items() if len(v) > 5}
    return _round({
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
        "timeframes": tfs,
    })
