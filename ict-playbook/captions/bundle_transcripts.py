#!/usr/bin/env python3
"""Merge transcripts into numbered Markdown bundles for upload to a Claude Project.

    python bundle_transcripts.py --src ./transcripts --out ./bundles --max-chars 400000
"""
import argparse
from pathlib import Path


def read_transcript(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    head, _, body = raw.partition("\n\n")
    title = next((l[7:] for l in head.splitlines() if l.startswith("TITLE: ")), path.stem)
    return title, body.strip()


def build_bundles(src: Path, min_chars: int, max_chars: int) -> list[list[tuple[str, str]]]:
    seen, docs = set(), []
    for p in sorted(src.glob("*_*_*.txt")):  # filenames start with upload date
        title, body = read_transcript(p)
        key = hash(body)
        if len(body) < min_chars or key in seen:
            continue
        seen.add(key)
        docs.append((title, body))
    bundles, cur, size = [], [], 0
    for title, body in docs:
        if cur and size + len(body) > max_chars:
            bundles.append(cur)
            cur, size = [], 0
        cur.append((title, body))
        size += len(body) + len(title) + 20
    if cur:
        bundles.append(cur)
    return bundles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("transcripts"))
    ap.add_argument("--out", type=Path, default=Path("bundles"))
    ap.add_argument("--max-chars", type=int, default=400_000)
    ap.add_argument("--min-chars", type=int, default=500, help="drop near-empty transcripts")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundles = build_bundles(args.src, args.min_chars, args.max_chars)
    index = ["# Transcript index\n"]
    for n, docs in enumerate(bundles, 1):
        name = f"bundle_{n:02d}.md"
        parts = [f"# Bundle {n}\n"]
        index.append(f"\n## {name}\n")
        for title, body in docs:
            parts.append(f"\n## {title}\n\n{body}\n")
            index.append(f"- {title}")
        (args.out / name).write_text("".join(parts), encoding="utf-8")
    (args.out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"{sum(len(b) for b in bundles)} transcripts -> {len(bundles)} bundles in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
