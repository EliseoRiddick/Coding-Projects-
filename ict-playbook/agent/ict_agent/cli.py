"""CLI: python -m ict_agent.cli --data-dir data --playbook playbook.md --symbol NQ"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .analyze import prepare, run_analysis
from .data import load_timeframes
from .images import image_blocks
from .risk import RiskConfig


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
    ap.add_argument("--equity", type=float, help="account equity; with --value-per-point turns on sizing and risk checks")
    ap.add_argument("--value-per-point", type=float, help="account-currency value of a 1.0 price move per unit (NQ 20, MNQ 2)")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="risk per trade in %% of equity (R93: 1 default, 0.5 beginner, 2 ceiling)")
    ap.add_argument("--size-step", type=float, default=1.0, help="smallest size increment (1 contract, 0.01 lot)")
    ap.add_argument("--journal", type=Path, help="JSON trade journal used for R95/R96 (see risk.log_trade)")
    ap.add_argument("--dry-run", action="store_true", help="print the assembled prompt; no API call")
    a = ap.parse_args(argv)

    frames = load_timeframes(a.data_dir, a.tz, [t for t in a.derive.split(",") if t])
    asof = pd.Timestamp(a.asof, tz=a.tz) if a.asof else None
    playbook = a.playbook.read_text(encoding="utf-8")
    econ = a.econ.read_text(encoding="utf-8") if a.econ else ""
    images = image_blocks(a.images)
    risk_cfg = None
    if a.equity or a.value_per_point:
        if not (a.equity and a.value_per_point):
            ap.error("--equity and --value-per-point must be given together")
        risk_cfg = RiskConfig(a.equity, a.value_per_point, a.risk_pct, a.size_step)

    if a.dry_run:
        features, system, content = prepare(frames, a.symbol, playbook, econ, images, asof, [p.name for p in a.images])
        print(system[0]["text"][:3000], "\n...\n")
        print(content[-1]["text"])
        return 0

    try:
        result, _ = run_analysis(frames, a.symbol, playbook, econ, images, asof, a.model, a.effort, a.out_dir,
                                 risk_cfg, a.journal, [p.name for p in a.images])
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {a.out_dir}\nEducational analysis only - not financial advice.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
