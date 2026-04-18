from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from . import mistral


@dataclass
class Chunk:
    doc: str
    page: int
    text: str
    idx: int
    headings: List[str] = field(default_factory=list)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")


def extract_pages(path: str, filename: str) -> List[Tuple[int, str]]:
    pages = mistral.ocr_file(path, filename)
    return [(int(p.get("index", i)) + 1, _rewrite_images(p.get("markdown", "") or ""))
            for i, p in enumerate(pages)]


def _rewrite_images(md: str) -> str:
    return _IMG.sub(lambda m: f"[Image: {(m.group(1) or 'figure').strip()}]", md)


def _split_blocks(md: str) -> List[Tuple[str, object]]:
    blocks: List[Tuple[str, object]] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(("text", text))
            buf.clear()

    for line in md.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            blocks.append(("heading", (len(m.group(1)), m.group(2).strip())))
        elif line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return blocks


def _hard_split(text: str, n: int) -> List[str]:
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(0, len(text), n)]


def chunk_pages(
    pages: List[Tuple[int, str]],
    doc: str,
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    headings: List[str] = []
    buf = ""

    for page, md in pages:
        if not md:
            continue
        for kind, payload in _split_blocks(md):
            if kind == "heading":
                level, title = payload  # type: ignore[assignment]
                if buf:
                    chunks.append(Chunk(doc, page, buf.strip(), idx, list(headings))); idx += 1
                    buf = ""
                headings = headings[: max(0, level - 1)] + [title]
                continue
            for piece in _hard_split(payload, target_chars):  # type: ignore[arg-type]
                if not buf or len(buf) + len(piece) + 2 <= target_chars:
                    buf = (buf + "\n\n" + piece).strip() if buf else piece
                else:
                    chunks.append(Chunk(doc, page, buf.strip(), idx, list(headings))); idx += 1
                    tail = buf[-overlap_chars:] if overlap_chars else ""
                    buf = (tail + "\n\n" + piece).strip() if tail else piece
        if buf:
            chunks.append(Chunk(doc, page, buf.strip(), idx, list(headings))); idx += 1
            buf = ""

    return chunks
