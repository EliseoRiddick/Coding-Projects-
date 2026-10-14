import pytest

from ict_agent import risk
from ict_agent.risk import RiskConfig, apply_guardrails, check_idea, log_trade, load_journal, risk_state

NQ = RiskConfig(equity=50_000, value_per_point=20)       # 1% = $500
MNQ = RiskConfig(equity=50_000, value_per_point=2)


def idea(**kw):
    base = {"bias": "long", "entry_zone": [18000, 18010], "stop": 17990, "targets": [18050, 18100]}
    return {**base, **kw}


def test_config_enforces_r93_ceiling_and_positivity():
    with pytest.raises(ValueError):
        RiskConfig(equity=50_000, value_per_point=20, risk_pct=2.5)
    with pytest.raises(ValueError):
        RiskConfig(equity=0, value_per_point=20)
    RiskConfig(equity=1, value_per_point=1, risk_pct=2.0)  # exactly the ceiling is allowed


def test_size_rounds_down_against_the_worst_fill():
    r = check_idea(idea(), NQ, 1.0)
    # long: widest stop is from the top of the zone (18010) to 17990 = 20 pts x $20 = $400 per contract; $500 budget -> 1.25 -> 1
    assert r["approved"] and r["size"] == 1 and r["risk_points"] == 20 and r["worst_case_entry"] == 18010
    assert r["risk_amount"] == 400 and r["risk_pct_of_equity"] == 0.8
    assert r["reward_risk_by_target"] == [2.0, 4.5]
    m = check_idea(idea(), MNQ, 1.0)                  # micros: $40 per contract -> 12.5 -> 12
    assert m["size"] == 12 and m["risk_amount"] == 480


def test_short_uses_bottom_of_zone():
    r = check_idea(idea(bias="short", stop=18030, targets=[17950]), MNQ, 1.0)
    assert r["worst_case_entry"] == 18000 and r["risk_points"] == 30 and r["size"] == 8


def test_never_exceeds_budget_for_many_random_inputs():
    import random
    rnd = random.Random(1)
    for _ in range(500):
        lo = rnd.uniform(100, 20000); hi = lo + rnd.uniform(0, 30); stop = lo - rnd.uniform(0.5, 400)
        cfg = RiskConfig(equity=rnd.uniform(1000, 500000), value_per_point=rnd.choice([0.5, 2, 20]), size_step=rnd.choice([1, 0.01]))
        pct = rnd.uniform(0.1, 2.0)
        r = check_idea(idea(entry_zone=[lo, hi], stop=stop, targets=[hi + 50]), cfg, pct)
        if r["approved"]:
            assert r["risk_amount"] <= cfg.equity * pct / 100 + 1e-6


def test_stop_too_wide_for_budget_is_rejected_with_micro_hint():
    r = check_idea(idea(stop=17900), NQ, 1.0)          # 110 pts x $20 = $2,200 > $500
    assert not r["approved"] and "R94" in r["issues"][0] and "micro" in r["issues"][0]
    assert check_idea(idea(stop=17900), MNQ, 1.0)["approved"]   # $220 per micro fits


@pytest.mark.parametrize("kw,fragment", [
    (dict(stop=18005), "below the entry zone"),                       # stop inside the zone
    (dict(stop=18020), "below the entry zone"),                       # stop on the wrong side
    (dict(targets=[18005]), "above the entry zone"),
    (dict(bias="short"), "above the entry zone"),
    (dict(bias="buy"), "bias"),
    (dict(entry_zone=[18000]), "entry_zone"),
    (dict(stop="17990"), "stop"),
    (dict(targets=[]), "targets"),
    (dict(stop=float("nan")), "stop"),
])
def test_bad_geometry_is_rejected(kw, fragment):
    r = check_idea(idea(**kw), NQ, 1.0)
    assert not r["approved"] and any(fragment in i for i in r["issues"]) and "size" not in r


def test_min_reward_risk_filter_is_optional():
    cfg = RiskConfig(equity=50_000, value_per_point=2, min_reward_risk=3.0)
    assert not check_idea(idea(), cfg, 1.0)["approved"]       # first target is 2R
    assert check_idea(idea(targets=[18100]), cfg, 1.0)["approved"]


def T(pnl, risk_amount=500, **kw):
    return {"pnl": pnl, "risk_amount": risk_amount, **kw}


def test_r95_full_loss_halves_then_recovery_restores():
    assert risk_state([], NQ)["effective_risk_pct"] == 1.0
    assert risk_state([T(-500)], NQ)["effective_risk_pct"] == 0.5
    assert risk_state([T(-500), T(-500)], NQ)["effective_risk_pct"] == 0.25          # second loss halves again
    assert risk_state([T(-500), T(200)], NQ)["effective_risk_pct"] == 0.5           # recovered 200 < half of 500
    assert risk_state([T(-500), T(300)], NQ)["effective_risk_pct"] == 1.0           # recovered >= half -> full
    s = risk_state([T(-500), T(100)], NQ)
    assert s["loss_to_recover"] == 400 and any("R95" in r for r in s["reasons"])


def test_partial_loss_does_not_halve():
    assert risk_state([T(-150)], NQ)["multiplier"] == 1.0


def test_five_wins_halve_only_the_next_trade():
    wins = [T(100)] * 5
    assert risk_state(wins, NQ)["effective_risk_pct"] == 0.5
    assert risk_state(wins + [T(100)], NQ)["effective_risk_pct"] == 1.0
    assert risk_state([T(100)] * 4, NQ)["effective_risk_pct"] == 1.0


def test_r96_never_size_up_after_a_loss():
    s = risk_state([T(-100, risk_pct=0.5)], NQ)        # partial loss, but the last trade only risked 0.5%
    assert s["effective_risk_pct"] == 0.5 and any("R96" in r for r in s["reasons"])


def test_journal_round_trip_and_state(tmp_path):
    p = tmp_path / "sub" / "journal.json"
    assert load_journal(p) == []
    log_trade(p, pnl=-500, risk_amount=500, symbol="NQ", risk_pct=1.0)
    log_trade(p, pnl=300, risk_amount=250, symbol="NQ")
    j = load_journal(p)
    assert [t["pnl"] for t in j] == [-500, 300] and j[0]["symbol"] == "NQ"
    assert risk_state(j, NQ)["effective_risk_pct"] == 1.0


def features(status):
    return {"time": {"trading_window": {"status": status}}, "current_price": 18005.0, "daily_atr14": 250.0}


def test_apply_guardrails_sizes_with_state_and_gates_by_time_window():
    res = {"ideas": [idea(), idea(stop=17900)]}
    out = apply_guardrails(res, features("tradable_window"), NQ, [T(-500)])
    assert out["risk_state"]["effective_risk_pct"] == 0.5
    a, b = out["ideas"][0]["risk"], out["ideas"][1]["risk"]
    # after a full-risk loss the budget is 0.5% = $250, but one NQ contract risks $400 -> rejected, never rounded up
    assert not a["approved"] and not a["executable_now"] and "R94" in a["issues"][0] and not b["approved"]
    ok = apply_guardrails({"ideas": [idea()]}, features("tradable_window"), MNQ, [T(-500)])["ideas"][0]["risk"]
    assert ok["approved"] and ok["executable_now"] and ok["size"] == 6 and ok["risk_amount"] == 240   # 6 micros <= $250
    res = {"ideas": [idea()]}
    r = apply_guardrails(res, features("no_new_trades"), MNQ, [])["ideas"][0]["risk"]
    assert r["approved"] and not r["executable_now"] and "R53" in r["wait_reason"]


def test_journal_is_optional_and_never_crashes_a_run(tmp_path):
    assert load_journal(None) == [] and load_journal(tmp_path / "missing.json") == []
    folder = tmp_path / "journal.json"
    folder.mkdir()                                       # the real-world failure: a FOLDER where the file should be
    assert load_journal(folder) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_journal(bad) == []
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"trades": 5}')
    assert load_journal(wrong) == []
    assert risk_state(load_journal(folder), NQ)["effective_risk_pct"] == 1.0


def test_log_trade_refuses_folders_and_never_overwrites_a_damaged_journal(tmp_path):
    folder = tmp_path / "journal.json"
    folder.mkdir()
    with pytest.raises(ValueError, match="folder"):
        log_trade(folder, -100, 500)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not a readable trade journal"):
        log_trade(bad, -100, 500)
    assert bad.read_text() == "{not json"                # untouched
    ok = tmp_path / "ok.json"
    log_trade(ok, -100, 500)
    assert len(load_journal(ok)) == 1
