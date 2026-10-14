"""Prepare chart screenshots for the API: validate, downscale, base64-encode."""
import base64
import io
from pathlib import Path

from PIL import Image

MAX_EDGE = 1568  # larger images are downscaled by the API anyway; do it here to save upload size
MAX_IMAGES = 8


def image_block(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    img.load()
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def image_blocks(items: list[bytes | Path | str]) -> list[dict]:
    if len(items) > MAX_IMAGES:
        raise ValueError(f"Attach at most {MAX_IMAGES} screenshots (got {len(items)})")
    return [image_block(i if isinstance(i, bytes) else Path(i).read_bytes()) for i in items]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_folder(path: Path | str, newest: int = 6) -> tuple[list[dict], list[str], list[str]]:
    """Read the `newest` most recently modified screenshots in a folder (e.g. a Google Drive folder).
    Returns (image blocks, file names in presentation order, skipped files). The chosen files are presented in
    file-name order so you can number them (1_daily.png, 2_15m.png). A missing folder gives no images."""
    folder = Path(path)
    if not folder.is_dir():
        return [], [], []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    chosen = sorted(sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:min(newest, MAX_IMAGES)], key=lambda p: p.name.lower())
    blocks, names, skipped = [], [], []
    for p in chosen:
        try:
            blocks.append(image_block(p.read_bytes()))
            names.append(p.name)
        except Exception:  # unreadable / corrupt / unsupported image
            skipped.append(p.name)
    return blocks, names, skipped
