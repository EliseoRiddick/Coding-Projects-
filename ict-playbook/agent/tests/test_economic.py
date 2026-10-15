import pandas as pd
import pytest

from ict_agent import economic as eco
from ict_agent.risk import RiskConfig, apply_guardrails

TZ = "America/New_York"


def ts(s):
    return pd.Timestamp(s, tz=TZ)


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


FF = """Title,Country,Date,Time,Impact,Forecast,Previous
Non-Farm Employment Change,USD,03-07-2025,8:30am,High,160K,143K
CPI m/m,USD,03-12-2025,8:30am,High,0.3%,0.5%
Unemployment Claims,USD,03-06-2025,8:30am,Medium,235K,242K
ECB Press Conference,EUR,03-06-2025,8:30am,High,,
Bank Holiday,USD,03-07-2025,All Day,Holiday,,
Revised Productivity,USD,03-06-2025,9:00am,Low,,
"""
TE = """Date,Time,Country,Event,Actual,Previous,Consensus,TE Forecast,Importance
2025-03-05,08:15 AM,United States,ADP Employment Change,,183K,140K,150K,2
2025-03-05,10:00 AM,United States,ISM Services PMI,,53.5,53.0,53.2,3
2025-03-05,02:00 PM,United States,Fed Interest Rate Decision,,4.5%,4.5%,4.5%,3
"""
DT = """DateTime,Currency,Impact,Event
2025-03-05T13:30:00Z,USD,★★★,CPI y/y
2025-03-05 09:45:00,USD,★★,Services PMI
2025-03-05 11:00:00,USD,★,Minor thing
"""
SIMPLE = """date,time,event,impact
2025-03-05,08:30,Trade Balance,high
2025-03-05,14:00,FOMC Statement,high
"""


def test_reads_the_common_export_layouts(tmp_path):
    for name, text, n in (("ff.csv", FF, 6), ("te.csv", TE, 3), ("dt.csv", DT, 3), ("simple.csv", SIMPLE, 2)):
        d = tmp_path / name[:-4]
        d.mkdir()
        write(d, name, text)
        cal = eco.load_calendar(d, TZ)
        assert cal.available and len(cal.df) == n and cal.bad_rows == 0, (name, cal.messages)
        assert str(cal.df["time"].dt.tz) == TZ
    te = eco.load_calendar(tmp_path / "te", TZ).df
    assert list(te["impact"]) == ["medium", "high", "high"] and te.loc[2, "time"] == ts("2025-03-05 14:00")
    assert te.loc[1, "forecast"] == "53.0" and te.loc[1, "previous"] == "53.5"
    dt = eco.load_calendar(tmp_path / "dt", TZ).df
    assert dt.loc[0, "time"] == ts("2025-03-05 08:30") and list(dt["impact"]) == ["high", "medium", "low"]   # the Z timestamp was UTC
    ff = eco.load_calendar(tmp_path / "ff", TZ).df
    hol = ff[ff["event"] == "Bank Holiday"].iloc[0]
    assert hol["all_day"] and hol["impact"] == "holiday" and ff[ff["event"] == "CPI m/m"].iloc[0]["time"] == ts("2025-03-12 08:30")


@pytest.mark.parametrize("raw,event,want", [("High", "x", "high"), ("3", "x", "high"), ("red", "x", "high"), ("***", "x", "high"),
                                            ("Medium", "x", "medium"), ("2", "x", "medium"), ("**", "x", "medium"),
                                            ("Low", "x", "low"), ("1", "x", "low"), ("*", "x", "low"), ("Holiday", "x", "holiday"),
                                            ("", "Nonfarm Payrolls", "high"), ("", "Initial Jobless Claims", "medium"),
                                            ("", "Some Auction", "low"), ("???", "CPI", "high")])
def test_impact_normalisation(raw, event, want):
    assert eco._impact(raw, event) == want


def test_bad_rows_are_counted_not_fatal(tmp_path):
    write(tmp_path, "c.csv", "date,time,event,impact\n2025-03-05,08:30,CPI,high\nnot a date,08:30,Oops,high\n2025-03-05,08:30,,high\n")
    cal = eco.load_calendar(tmp_path, TZ)
    assert len(cal.df) == 1 and cal.bad_rows == 2 and any("2 calendar row" in m for m in cal.messages)


def test_unrecognised_file_never_raises(tmp_path):
    write(tmp_path, "weird.csv", "foo,bar\n1,2\n")
    cal = eco.load_calendar(tmp_path, TZ)
    assert not cal.available and any("foo" in m and "weird.csv" in m for m in cal.messages)
    assert not eco.load_calendar(tmp_path / "missing", TZ).available
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not eco.load_calendar(empty, TZ).available
    out = eco.summarize(cal, ts("2025-03-05 10:00"))
    assert out["available"] is False and "could not be checked" in out["note"]


def test_two_files_are_merged_and_deduplicated(tmp_path):
    write(tmp_path, "a.csv", SIMPLE)
    write(tmp_path, "b.csv", SIMPLE + "2025-03-06,08:30,Jobless Claims,medium\n")
    cal = eco.load_calendar(tmp_path, TZ)
    assert len(cal.df) == 3 and cal.files == ["a.csv", "b.csv"]


def cal_from(tmp_path, text):
    write(tmp_path, "c.csv", text)
    return eco.load_calendar(tmp_path, TZ)


def test_r64_holds_the_release_candle_only(tmp_path):
    cal = cal_from(tmp_path, "date,time,currency,event,impact\n2025-03-12,08:30,USD,CPI m/m,high\n2025-03-12,08:30,EUR,German CPI,high\n"
                             "2025-03-12,11:00,USD,Minor,low\n")
    for when, held in (("2025-03-12 08:29", False), ("2025-03-12 08:30", True), ("2025-03-12 08:32", True), ("2025-03-12 08:33", False)):
        out = eco.summarize(cal, ts(when))
        assert (out["news_hold"] is not None) is held, when
    out = eco.summarize(cal, ts("2025-03-12 08:31"))
    assert out["news_hold"]["rule"] == "R64" and out["news_hold"]["until"].startswith("2025-03-12T08:33")
    f = out["flags"]
    assert f["data_day_0830"] and f["high_impact_today"] == ["08:30 CPI m/m"]
    assert [e["event"] for e in out["events"]] == ["CPI m/m"]                  # EUR and low-impact rows dropped
    assert eco.summarize(cal, ts("2025-03-12 08:00"))["next_high_impact"]["minutes_from_now"] == 30


def test_r65_fomc_decision_holds_thirty_minutes_and_ignores_minutes_and_speeches(tmp_path):
    cal = cal_from(tmp_path, "date,time,event,impact\n2025-03-19,14:00,Fed Interest Rate Decision,high\n2025-03-19,10:00,FOMC Member Williams Speaks,high\n"
                             "2025-03-19,14:30,FOMC Press Conference,high\n2025-03-19,15:00,FOMC Meeting Minutes,high\n")
    out = eco.summarize(cal, ts("2025-03-19 14:10"))
    assert out["news_hold"]["rule"] == "R65" and out["flags"]["fomc_today"] and "14:00" in out["news_hold"]["reason"]
    assert eco.summarize(cal, ts("2025-03-19 14:29"))["news_hold"]["rule"] == "R65"
    assert eco.summarize(cal, ts("2025-03-19 14:31"))["news_hold"]["rule"] == "R64"   # the press conference candle
    assert eco.summarize(cal, ts("2025-03-19 14:40"))["news_hold"] is None
    assert eco.summarize(cal, ts("2025-03-19 10:30"))["flags"]["fomc_today"] is True   # decision is later today


def test_r66_nfp_week_stands_aside_from_wednesday_1100_until_release_is_out(tmp_path):
    cal = cal_from(tmp_path, "date,time,event,impact\n2025-03-07,08:30,Non-Farm Payrolls,high\n")
    for when, rule in (("2025-03-03 10:00", None), ("2025-03-05 10:59", None), ("2025-03-05 11:00", "R66"), ("2025-03-06 15:00", "R66"),
                       ("2025-03-07 07:00", "R66"), ("2025-03-07 08:32", "R66"), ("2025-03-07 08:33", None), ("2025-03-07 09:00", None),
                       ("2025-03-10 11:00", None)):
        h = eco.summarize(cal, ts(when))["news_hold"]
        assert (h["rule"] if h else None) == rule, when
    assert eco.summarize(cal, ts("2025-03-04 12:00"))["flags"]["nfp_week"] is True
    assert eco.summarize(cal, ts("2025-03-10 12:00"))["flags"]["nfp_week"] is False


def test_holiday_week_flag_and_stale_calendar_warning(tmp_path):
    cal = cal_from(tmp_path, FF)
    f = eco.summarize(cal, ts("2025-03-06 12:00"))["flags"]
    assert f["holiday_week"] and f["holiday_names"] == ["Bank Holiday"]
    assert not eco.summarize(cal, ts("2025-03-12 12:00"))["flags"]["holiday_week"]
    stale = eco.summarize(cal, ts("2025-04-01 12:00"))
    assert any("before today" in w for w in stale["warnings"]) and any("out of date" in w for w in stale["warnings"])


def test_event_list_is_capped_and_near_first(tmp_path):
    rows = "\n".join(f"2025-03-{d:02d},{h:02d}:00,Event {d}-{h},high" for d in range(3, 10) for h in range(7, 18))
    out = eco.summarize(cal_from(tmp_path, "date,time,event,impact\n" + rows), ts("2025-03-05 12:00"))
    assert len(out["events"]) == eco.MAX_EVENTS
    times = [e["time"] for e in out["events"]]
    assert times == sorted(times) and any(t.startswith("2025-03-05") for t in times)


def test_news_hold_blocks_executable_now_and_missing_calendar_is_flagged():
    idea = {"bias": "long", "entry_zone": [18000, 18010], "stop": 17990, "targets": [18050]}
    base = {"symbol": "NQ", "current_price": 18005.0, "daily_atr14": 250.0,
            "time": {"trading_window": {"status": "tradable_window"}}}
    cfg = RiskConfig(equity=50000, value_per_point=2)
    hold = {"rule": "R64", "reason": "CPI released at 08:30", "until": "x"}
    r = apply_guardrails({"ideas": [dict(idea)]}, {**base, "economic": {"available": True, "news_hold": hold}}, cfg, [])["ideas"][0]["risk"]
    assert r["approved"] and not r["executable_now"] and r["wait_reason"] == "R64: CPI released at 08:30"
    r = apply_guardrails({"ideas": [dict(idea)]}, {**base, "economic": {"available": True, "news_hold": None}}, cfg, [])["ideas"][0]["risk"]
    assert r["executable_now"]
    r = apply_guardrails({"ideas": [dict(idea)]}, {**base, "economic": {"available": False}}, cfg, [])["ideas"][0]["risk"]
    assert r["executable_now"] and any("NOT checked" in w for w in r["warnings"])
