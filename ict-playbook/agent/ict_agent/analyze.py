"""One entry point used by both the CLI and the Colab notebook."""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import pandas as pd

from . import economic as econ_mod
from . import risk as risk_mod
from .features import build_features
from .prompt import parse_json_response, system_blocks, user_content


def prepare(frames, symbol, playbook, econ="", images=None, asof=None, image_names=None, calendar=None):
    """Compute features and assemble the request pieces (no API call). `calendar` is an economic.Calendar."""
    features = build_features(frames, symbol, asof)
    features["economic"] = econ_mod.summarize(calendar or econ_mod.Calendar(None, messages=["No economic calendar was provided."]),
                                              pd.Timestamp(features["time"]["asof"]))
    return features, system_blocks(playbook), user_content(features, econ, images, image_names)


def run_analysis(frames, symbol, playbook, econ="", images=None, asof=None,
                 model="claude-opus-5-5", effort="high", out_dir: Path | None = None,
                 risk_cfg: "risk_mod.RiskConfig | None" = None, journal_path: Path | str | None = None,
                 image_names: list[str] | None = None, calendar=None):
    """Call Claude and return (result_dict, features). If risk_cfg is given, every idea is checked and sized in
    Python (risk.apply_guardrails) before anything is returned or saved. Saves JSON under out_dir if given."""
    import anthropic  # late import: tests and --dry-run need no API setup

    features, system, content = prepare(frames, symbol, playbook, econ, images, asof, image_names, calendar)
    # Keys that are not tied to one workspace must say which workspace to use (set ANTHROPIC_WORKSPACE_ID).
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=system,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        resp = stream.get_final_message()
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Model stopped early: {resp.stop_reason}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        result = parse_json_response(text)
    except ValueError:
        result = {"raw": text}
    result["_meta"] = {"symbol": symbol, "asof": features["time"]["asof"], "model": model,
                       "images": len(images or []), "cache_read_tokens": resp.usage.cache_read_input_tokens,
                       "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    out_dir = Path(out_dir) if out_dir else None
    stamp = f"{symbol}_{datetime.now():%Y%m%d_%H%M%S}"
    if out_dir:  # keep the paid answer even if a later step fails
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stamp}_raw.json").write_text(json.dumps(result, indent=2))
    if risk_cfg is not None:
        try:
            risk_mod.apply_guardrails(result, features, risk_cfg, risk_mod.load_journal(journal_path))
        except Exception as e:  # fail closed: never show ideas that were not risk-checked
            result["risk_state"], result["risk_error"] = None, f"{type(e).__name__}: {e}"
            for idea in result.get("ideas", []):
                if isinstance(idea, dict):
                    idea["risk"] = {"approved": False, "executable_now": False, "warnings": [],
                                    "issues": [f"risk checks could not run: {result['risk_error']}"]}
    if out_dir:
        (out_dir / f"{stamp}.json").write_text(json.dumps(result, indent=2))
    return result, features
