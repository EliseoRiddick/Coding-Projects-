"""Turn the model's ideas + Python's risk checks into one short card, and a full report underneath."""

CONFIDENCE = {"high": 3, "medium": 2, "low": 1}
WAIT_TEXT = {"no_new_trades": "R53: no new trades 11:30-13:30", "observe_only": "R52: 7:00-9:00 is observation only",
             "closed_or_conditional_only": "market closed / conditional scenarios only"}


def fmt_price(x) -> str:
    """Readable price: 2 decimals normally, 5 for small prices (forex). Non-numbers pass through."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    return f"{x:.5f}" if abs(x) < 10 else f"{x:.2f}"


def _rank(idea: dict):
    rr = (idea.get("risk", {}).get("reward_risk_by_target") or [0])[0]
    return CONFIDENCE.get(str(idea.get("confidence", "")).lower(), 0), rr


def make_card(result: dict, features: dict) -> dict:
    """Pick one idea. SETUP = approved by the risk checks and inside a trading window (the idea's own
    'wait for' conditions still have to happen); WAIT = approved but the window is closed; NO TRADE otherwise."""
    ideas = [i for i in result.get("ideas", []) if isinstance(i, dict) and isinstance(i.get("risk"), dict)]
    approved = [i for i in ideas if i["risk"].get("approved")]
    live = [i for i in approved if i["risk"].get("executable_now")]
    base = {"symbol": features["symbol"], "asof": features["time"]["asof"], "price": features["current_price"],
            "window": features["time"]["trading_window"]["status"], "risk_state": result.get("risk_state", {}),
            "ideas_considered": len(ideas)}
    pick = max(live, key=_rank) if live else max(approved, key=_rank) if approved else None
    if pick is None:
        reasons = [str(r) for r in result.get("no_trade_reasons", [])]
        for i in ideas:
            reasons += [f"idea '{i.get('setup', i.get('bias', '?'))}' rejected: {x}" for x in i["risk"].get("issues", [])]
        return {**base, "decision": "NO TRADE", "reasons": reasons or ["The model found no setup that fits the playbook."]}
    r = pick["risk"]
    lo, hi = sorted(pick["entry_zone"])
    return {**base, "decision": "SETUP" if r.get("executable_now") else "WAIT",
            "direction": "LONG (buy)" if pick["bias"] == "long" else "SHORT (sell)",
            "entry_zone": [lo, hi], "stop": pick["stop"], "take_profit": list(pick["targets"]),
            "risk_points": r["risk_points"], "size": r["size"], "risk_amount": r["risk_amount"], "risk_pct": r["risk_pct_of_equity"],
            "reward_risk": r["reward_risk_by_target"], "confidence": pick.get("confidence"),
            "playbook_refs": pick.get("playbook_refs", []), "wait_for": pick.get("conditions_to_wait_for", []),
            "invalidation": pick.get("invalidation", ""), "why": pick.get("rationale", ""),
            "wait_reason": r.get("wait_reason"), "warnings": r.get("warnings", []),
            "other_approved_ideas": len(approved) - 1}


def headline(card: dict) -> str:
    """One plain-English sentence a beginner can act on."""
    if card["decision"] == "NO TRADE":
        why = card["reasons"][0] if card["reasons"] else "nothing fits the playbook"
        return f"🚫 **No trade right now.** {why}"
    word = "BUY" if card["direction"].startswith("LONG") else "SELL"
    lo, hi = card["entry_zone"]
    tps = card["take_profit"]
    target = f"{fmt_price(tps[0])}" + (f", then {fmt_price(tps[1])}" if len(tps) > 1 else "")
    plan = (f"**{word} {card['symbol']}.** Enter between {fmt_price(lo)} and {fmt_price(hi)}. "
            f"Stop loss at {fmt_price(card['stop'])} (about {fmt_price(card['risk_points'])} points of risk). Take profit at {target}.")
    if card["decision"] == "WAIT":
        return f"⏳ **Valid idea, but not now:** {plan} {card['wait_reason'] or 'The trading window is closed.'}"
    return f"✅ **Trade idea:** {plan} Only enter once the conditions below have happened."


def card_markdown(card: dict) -> str:
    head = f"## {card['symbol']} - {card['decision']}   *(price {fmt_price(card['price'])}, {card['asof']})*\n\n> {headline(card)}"
    if card["decision"] == "NO TRADE":
        return head + "\n\n" + ("**Other reasons:** " + "; ".join(card["reasons"][1:]) if len(card["reasons"]) > 1 else "")
    lines = [head, "",
             "| | |", "|---|---|",
             f"| **Direction** | **{card['direction']}** |",
             f"| **Entry zone** | {fmt_price(card['entry_zone'][0])} - {fmt_price(card['entry_zone'][1])} |",
             f"| **Stop loss** | {fmt_price(card['stop'])} |",
             f"| **Take profit** | {' / '.join(map(fmt_price, card['take_profit']))}  (reward:risk {' / '.join(map(str, card['reward_risk']))}) |",
             f"| **Size** | {card['size']:g} units - risks {card['risk_amount']:,.2f} ({card['risk_pct']}% of equity) |",
             f"| **Confidence** | {card['confidence']}  (playbook: {', '.join(map(str, card['playbook_refs']))}) |", ""]
    lines += ["*Entry zone = the price range to buy/sell in. Stop loss = where you exit if you are wrong. Take profit = where you take gains.*", ""]
    if card["decision"] == "WAIT":
        lines.append(f"**Not now:** {card['wait_reason'] or 'outside the playbook trading windows'}. The setup is valid for the next allowed window.")
    else:
        lines.append("**In an allowed trading window.** Enter only after these happen:")
    lines += [f"- {c}" for c in card["wait_for"]]
    lines += [f"\n**Invalidation:** {card['invalidation']}", f"\n**Why:** {card['why']}"]
    lines += [f"\n*warning:* {w}" for w in card["warnings"]]
    if card["other_approved_ideas"]:
        lines.append(f"\n*{card['other_approved_ideas']} other approved idea(s) are in the full report below.*")
    return "\n".join(lines)


def report_markdown(result: dict, features: dict, saved_to: str = "") -> str:
    """Everything the model said plus the code-checked sizing, for reading under the card."""
    t = features["time"]
    out = [f"### Full report - {features['symbol']} {t['asof']} ({t['day_of_week']}, {t['current_session']})",
           f"**Market read:** {result.get('market_read', result.get('raw', ''))}",
           f"**Session note:** {result.get('session_note', '')}"]
    for n, idea in enumerate(result.get("ideas", []), 1):
        out.append(f"\n#### Idea {n}: {str(idea.get('bias', '?')).upper()} - {idea.get('setup', '')}  *(confidence: {idea.get('confidence', '?')})*")
        out.append(f"- **Entry zone:** {' - '.join(map(fmt_price, idea.get('entry_zone') or []))}   **Stop:** {fmt_price(idea.get('stop'))}   "
                   f"**Targets:** {', '.join(map(fmt_price, idea.get('targets') or []))}")
        out.append("- **Wait for:** " + "; ".join(map(str, idea.get("conditions_to_wait_for") or [])))
        out.append(f"- **Invalidation:** {idea.get('invalidation', '')}")
        out.append(f"- **Why:** {idea.get('rationale', '')}  *(playbook: {', '.join(map(str, idea.get('playbook_refs') or []))})*")
        r = idea.get("risk", {})
        if r.get("approved"):
            now = "executable now" if r.get("executable_now") else f"not now - {r.get('wait_reason', 'outside trading window')}"
            out.append(f"- **Size (code-checked):** {r['size']:g} units, risk {r['risk_amount']:,.2f} ({r['risk_pct_of_equity']}% of equity), "
                       f"stop {fmt_price(r['risk_points'])} pts from the worst-case fill {fmt_price(r['worst_case_entry'])}, reward:risk {r['reward_risk_by_target']} - {now}")
        else:
            out.append("- **REJECTED by risk checks:** " + "; ".join(r.get("issues", ["no risk data"])))
        out += [f"- *warning:* {w}" for w in r.get("warnings", [])]
    if not result.get("ideas"):
        out.append("\n**No trade.** " + "; ".join(map(str, result.get("no_trade_reasons", []))))
    rs = result.get("risk_state")
    if rs:
        out.append(f"\n**Risk state:** {rs['effective_risk_pct']}% per trade (base {rs['base_risk_pct']}%). " + " ".join(rs["reasons"]))
    m = result.get("_meta", {})
    if m:
        out.append(f"\n*tokens: {m['input_tokens']:,} in ({m['cache_read_tokens']:,} cached) / {m['output_tokens']:,} out."
                   f"{' Saved to ' + saved_to + '.' if saved_to else ''} Educational analysis only - not financial advice.*")
    return "\n".join(out)
