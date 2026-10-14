"""Hard risk guardrails, enforced in Python AFTER the model answers (playbook R93-R96).

The model proposes entry/stop/targets; this module validates the geometry, computes the position size and
applies the loss/win-streak risk rules from your trade journal. Nothing here is delegated to the model.

    R93  risk <= 1% of equity per trade (0.5% beginners); 2% is the absolute ceiling
    R94  size = (equity x risk%) / (stop distance x value per unit), rounded DOWN
    R95  after a full-risk loss halve risk; back to full only after recovering half the loss;
         halve again after a second loss and after five consecutive wins
    R96  never increase size to recover losses
"""
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ABS_CEILING_PCT = 2.0         # R93
HALVE_AFTER_WINS = 5          # R95
FULL_RISK_LOSS_FRACTION = 0.9  # a loss of >= 90% of the planned risk counts as a "full-risk loss" (ASSUMPTION)
FAR_ENTRY_DAILY_ATR = 3.0     # warn when an entry zone is this many daily ATRs from price (ASSUMPTION)


@dataclass
class RiskConfig:
    equity: float
    value_per_point: float      # account-currency value of a 1.0 price move for ONE unit (NQ 20, MNQ 2, ...)
    risk_pct: float = 1.0       # R93 default; use 0.5 while learning
    size_step: float = 1.0      # smallest tradable increment: 1 contract, 0.01 lot, ...
    min_reward_risk: float | None = None  # optional extra filter on the first target; the playbook sets none

    def __post_init__(self):
        if self.equity <= 0 or self.value_per_point <= 0 or self.size_step <= 0:
            raise ValueError("equity, value_per_point and size_step must be positive")
        if not 0 < self.risk_pct <= ABS_CEILING_PCT:
            raise ValueError(f"risk_pct must be above 0 and at most {ABS_CEILING_PCT}% (R93 absolute ceiling)")


# ------------------------------------------------------------------ journal (R95/R96)
def _read_journal(path: Path) -> list[dict]:
    """Strict read: raises ValueError for a folder or a file that is not a valid journal."""
    if path.is_dir():
        raise ValueError(f"{path} is a folder, not a file - choose a file name such as trades.json")
    try:
        data = json.loads(path.read_text())
        trades = data["trades"]
        if not isinstance(trades, list):
            raise TypeError("'trades' is not a list")
        return trades
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} is not a readable trade journal ({type(e).__name__}: {e})") from e


def load_journal(path: Path | str | None) -> list[dict]:
    """The journal is optional: a missing, unreadable or malformed file simply means 'no history' (never raises)."""
    if not path or not Path(path).exists():
        return []
    try:
        return _read_journal(Path(path))
    except ValueError:
        return []


def log_trade(path: Path | str, pnl: float, risk_amount: float, symbol: str = "", risk_pct: float | None = None,
              date: str | None = None) -> None:
    """Append one CLOSED trade: pnl (negative for a loss) and the amount you planned to risk on it.
    Refuses to touch a folder or a damaged file rather than overwrite it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    trades = _read_journal(p) if p.exists() else []
    trades.append({"date": date or datetime.now().isoformat(timespec="minutes"), "symbol": symbol,
                   "pnl": float(pnl), "risk_amount": float(risk_amount), "risk_pct": risk_pct})
    p.write_text(json.dumps({"trades": trades}, indent=1))


def risk_state(trades: list[dict], cfg: RiskConfig) -> dict:
    """Replay the journal to get the risk multiplier for the NEXT trade.

    Interpretation of R95 (the rule is terse): a full-risk loss halves the multiplier; each further full-risk loss
    halves again; it returns to 1.0 once cumulative wins recover half of the cumulative losses; five consecutive
    wins halve risk for the NEXT trade only. R96 is enforced separately by capping risk at the last loss's risk."""
    m, wins, loss_total, recovered, streak_flag = 1.0, 0, 0.0, 0.0, False
    reasons: list[str] = []
    last = None
    for t in trades:
        streak_flag = False  # a streak halving only applies to the trade right after the fifth win
        pnl, risk = float(t["pnl"]), float(t["risk_amount"])
        if pnl < 0:
            wins = 0
            loss_total += -pnl
            if -pnl >= FULL_RISK_LOSS_FRACTION * risk:
                m /= 2
        elif pnl > 0:
            wins += 1
            if loss_total > 0:
                recovered += pnl
                if recovered >= 0.5 * loss_total:
                    m, loss_total, recovered = 1.0, 0.0, 0.0
            if wins >= HALVE_AFTER_WINS:
                streak_flag, wins = True, 0
        last = t
    mult = m * (0.5 if streak_flag else 1.0)
    if m < 1.0:
        reasons.append(f"R95: risk is x{m:g} after full-risk loss(es); back to full after recovering half of "
                       f"{loss_total:.2f} lost (recovered {recovered:.2f})")
    if streak_flag:
        reasons.append(f"R95: {HALVE_AFTER_WINS} consecutive wins - next trade at half risk")
    eff = cfg.risk_pct * mult
    cap = None
    if last and float(last["pnl"]) < 0 and last.get("risk_pct"):
        cap = float(last["risk_pct"])
        if eff > cap:
            eff = cap
            reasons.append(f"R96: capped at {cap:g}% (the last trade, a loss, risked that much); never size up to recover")
    return {"base_risk_pct": cfg.risk_pct, "multiplier": mult, "effective_risk_pct": round(eff, 4),
            "loss_to_recover": round(max(loss_total - recovered, 0.0), 2), "trades_in_journal": len(trades),
            "reasons": reasons}


# ------------------------------------------------------------------ per-idea checks (R93/R94)
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_idea(idea: dict, cfg: RiskConfig, eff_risk_pct: float, price: float | None = None,
               daily_atr: float | None = None) -> dict:
    issues, warnings = [], []
    bias = idea.get("bias")
    zone, stop, targets = idea.get("entry_zone"), idea.get("stop"), idea.get("targets")
    ok_zone = isinstance(zone, (list, tuple)) and len(zone) == 2 and all(_num(z) and z > 0 for z in zone)
    ok_targets = isinstance(targets, (list, tuple)) and targets and all(_num(t) and t > 0 for t in targets)
    if bias not in ("long", "short"):
        issues.append("bias must be 'long' or 'short'")
    if not ok_zone:
        issues.append("entry_zone must be two positive numbers")
    if not (_num(stop) and stop > 0):
        issues.append("stop must be a positive number")
    if not ok_targets:
        issues.append("targets must be a non-empty list of positive numbers")
    if issues:
        return {"approved": False, "issues": issues, "warnings": warnings}

    lo, hi = sorted(zone)
    long_ = bias == "long"
    if long_ and not stop < lo:
        issues.append("stop must be below the entry zone for a long")
    if not long_ and not stop > hi:
        issues.append("stop must be above the entry zone for a short")
    if long_ and any(t <= hi for t in targets):
        issues.append("every target must be above the entry zone for a long")
    if not long_ and any(t >= lo for t in targets):
        issues.append("every target must be below the entry zone for a short")
    if issues:
        return {"approved": False, "issues": issues, "warnings": warnings}

    worst = hi if long_ else lo            # the fill that makes the stop widest - size against it
    risk_pts = abs(worst - stop)
    per_unit = risk_pts * cfg.value_per_point
    budget = cfg.equity * eff_risk_pct / 100
    size = math.floor(budget / per_unit / cfg.size_step + 1e-9) * cfg.size_step
    size = round(size, 8)
    rr = [round(abs(t - worst) / risk_pts, 2) for t in targets]
    if size < cfg.size_step:
        min_risk_pct = per_unit * cfg.size_step / cfg.equity * 100
        issues.append(f"R94: stop is too wide for the risk budget - the smallest size ({cfg.size_step:g}) would risk "
                      f"{min_risk_pct:.2f}% of equity vs {eff_risk_pct:g}% allowed. Use a micro/smaller instrument or skip.")
    actual = size * per_unit
    pct = actual / cfg.equity * 100
    if pct > ABS_CEILING_PCT + 1e-9 or pct > eff_risk_pct + 1e-9:  # defence in depth; flooring should make this impossible
        issues.append(f"R93: computed risk {pct:.2f}% exceeds the allowed {eff_risk_pct:g}%")
    if cfg.min_reward_risk is not None and rr[0] < cfg.min_reward_risk:
        issues.append(f"first target reward:risk {rr[0]} is below your minimum {cfg.min_reward_risk}")
    if price and daily_atr and daily_atr > 0:
        dist = (lo - price) if lo > price else (price - hi) if hi < price else 0.0
        if dist / daily_atr > FAR_ENTRY_DAILY_ATR:
            warnings.append(f"entry zone is {dist / daily_atr:.1f} daily ATRs from the current price")
    return {"approved": not issues, "issues": issues, "warnings": warnings, "size": size,
            "risk_amount": round(actual, 2), "risk_pct_of_equity": round(pct, 3), "risk_points": round(risk_pts, 5),
            "worst_case_entry": worst, "reward_risk_by_target": rr}


def apply_guardrails(result: dict, features: dict, cfg: RiskConfig, trades: list[dict]) -> dict:
    """Attach `risk` to every idea and a `risk_state` summary to the result. Also enforces the time rules
    (R52/R53) as `executable_now`. Mutates and returns result."""
    state = risk_state(trades, cfg)
    status = features["time"]["trading_window"]["status"]
    price, atr = features.get("current_price"), features.get("daily_atr14")
    for idea in result.get("ideas", []):
        r = check_idea(idea, cfg, state["effective_risk_pct"], price, atr)
        r["executable_now"] = bool(r["approved"] and status == "tradable_window")
        if r["approved"] and status != "tradable_window":
            r["wait_reason"] = {"no_new_trades": "R53: no new trades 11:30-13:30",
                                "observe_only": "R52: 7:00-9:00 is observation only",
                                "closed_or_conditional_only": "market closed / conditional scenarios only"}.get(
                status, "R52: outside the predefined trading windows")
        idea["risk"] = r
    result["risk_state"] = state
    return result
