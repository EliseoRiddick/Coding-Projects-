"""CLI: python -m ict_agent.cli --data-dir data --playbook playbook.md --symbol NQ"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import load_timeframes
from .features import build_features
from .prompt import parse_json_response, system_blocks, user_message


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate trade ideas from multi-timeframe OHLC + a playbook")
    ap.add_argument("--data-dir", type=Path, required=True, help="folder with 1m.csv, 15m.csv, 1h.csv, 4h.csv, 1d.csv ...")
    ap.add_argument("--playbook", type=Path, required=True)
    ap.add_argument("--symbol", default="UNKNOWN")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--asof", help="evaluate as of this local time (YYYY-MM-DD HH:MM); default: last bar")
    ap.add_argument("--derive", default="", help="comma list of timeframes to resample from the finest, e.g. 1h,4h,1d")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--out-dir", type=Path, default=Path("ideas"))
    ap.add_argument("--dry-run", action="store_true", help="print the assembled prompt; no API call")
    a = ap.parse_args(argv)

    frames = load_timeframes(a.data_dir, a.tz, [t for t in a.derive.split(",") if t])
    asof = pd.Timestamp(a.asof, tz=a.tz) if a.asof else None
    features = build_features(frames, a.symbol, asof)
    playbook = a.playbook.read_text(encoding="utf-8")

    if a.dry_run:
        print(system_blocks(playbook)[0]["text"][:3000], "\n...\n")
        print(user_message(features))
        return 0

    import anthropic  # imported late so --dry-run and tests need no API setup

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or `ant auth login` profile
    with client.messages.stream(
        model=a.model,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={"effort": a.effort},
        system=system_blocks(playbook),
        messages=[{"role": "user", "content": user_message(features)}],
    ) as stream:
        resp = stream.get_final_message()

    if resp.stop_reason in ("refusal", "max_tokens"):
        print(f"Model stopped early: {resp.stop_reason}", file=sys.stderr)
        return 1
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        result = parse_json_response(text)
    except ValueError:
        result = {"raw": text}
    result["_meta"] = {"symbol": a.symbol, "asof": features["time"]["asof"], "model": a.model,
                       "cache_read_tokens": resp.usage.cache_read_input_tokens}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"{a.symbol}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved {path}\nEducational analysis only - not financial advice.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
