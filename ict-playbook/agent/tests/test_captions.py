from vtt_clean import clean_vtt
from bundle_transcripts import build_bundles

VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000 align:start position:0%
liquidity sits<00:00:01.000><c> above</c><00:00:01.500><c> highs</c>

00:00:02.000 --> 00:00:04.000
liquidity sits above highs

00:00:02.000 --> 00:00:04.000
liquidity sits above highs
and gets taken

00:00:04.000 --> 00:00:06.000
and gets taken
then reverses &amp; runs
"""


def test_clean_vtt_collapses_rolling_duplicates():
    out = clean_vtt(VTT)
    assert out == "liquidity sits above highs and gets taken then reverses & runs"


def test_bundles_split_and_dedupe(tmp_path):
    for i in range(3):
        (tmp_path / f"2020010{i}_vid{i}_t.txt").write_text(f"TITLE: T{i}\nURL: u\nDATE: x\n\n" + ("word " * 200) + str(i))
    (tmp_path / "20200109_dup_t.txt").write_text("TITLE: D\nURL: u\nDATE: x\n\n" + ("word " * 200) + "0")
    (tmp_path / "20200110_tiny_t.txt").write_text("TITLE: tiny\nURL: u\nDATE: x\n\nhi")
    bundles = build_bundles(tmp_path, min_chars=100, max_chars=2100)
    titles = [t for b in bundles for t, _ in b]
    assert titles == ["T0", "T1", "T2"] and len(bundles) == 2
