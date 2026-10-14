"""Convert YouTube WebVTT caption files into clean prose."""
import html
import re

_TAG = re.compile(r"<[^>]+>")
_TIMING = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s+-->")


def clean_vtt(text: str) -> str:
    """Strip headers/timestamps/tags and collapse auto-caption rolling duplicates."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if _TIMING.match(line) or line.isdigit():
            continue
        line = html.unescape(_TAG.sub("", line)).strip()
        if not line:
            continue
        # Auto-captions repeat each line across consecutive cues.
        if lines and (line == lines[-1] or lines[-1].endswith(line)):
            continue
        if lines and line.startswith(lines[-1]):
            lines[-1] = line
            continue
        lines.append(line)
    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).replace(">>", "\n\n").strip()
    return out
