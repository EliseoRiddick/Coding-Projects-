"""CLI: python -m ict_agent.cli --data-dir data --playbook playbook.md --symbol NQ"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .analyze import prepare, run_analysis
from .data import load_timeframes
from .images import image_blocks


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
    ap.add_argument("--images", type=Path, nargs="*", default=[], help="chart screenshots (png/jpg)")
    ap.add_argument("--econ", type=Path, help="text file with economic data / calendar notes")
    ap.add_argument("--dry-run", action="store_true", help="print the assembled prompt; no API call")
    a = ap.parse_args(argv)

    frames = load_timeframes(a.data_dir, a.tz, [t for t in a.derive.split(",") if t])
    asof = pd.Timestamp(a.asof, tz=a.tz) if a.asof else None
    playbook = a.playbook.read_text(encoding="utf-8")
    econ = a.econ.read_text(encoding="utf-8") if a.econ else ""
    images = image_blocks(a.images)

    if a.dry_run:
        features, system, content = prepare(frames, a.symbol, playbook, econ, images, asof)
        print(system[0]["text"][:3000], "\n...\n")
        print(content[-1]["text"])
        return 0

    try:
        result, _ = run_analysis(frames, a.symbol, playbook, econ, images, asof, a.model, a.effort, a.out_dir)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {a.out_dir}\nEducational analysis only - not financial advice.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
