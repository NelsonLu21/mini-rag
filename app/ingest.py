from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import List, Tuple
from . import mistral


_META_SYS = """Extract structured metadata from the opening of a document. Output JSON only, matching this schema:
{"title": string, "authors": [string, ...], "date": string|null, "summary": string}

Rules:
- Only include information literally present in the given text. Do not infer or guess.
- "title" is the document's own title (not a section heading). If not present, use an empty string.
- "authors" is a list of person names credited as authors. Empty list if absent.
- "date" is a publication or creation date in ISO form (YYYY-MM-DD or YYYY-MM or YYYY) if present, else null.
- "summary" is ONE sentence (max 30 words) describing what the document is about. If the text is too short to summarize, use an empty string.
"""


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


def extract_doc_meta(pages: List[Tuple[int, str]], filename: str) -> dict:
    head = "\n\n".join(md for _, md in pages[:2] if md).strip()
    empty = {"title": filename, "authors": [], "date": None, "summary": ""}
    if not head:
        return empty
    try:
        raw = mistral.chat(
            [{"role": "system", "content": _META_SYS},
             {"role": "user", "content": head[:6000]}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    title = str(data.get("title") or "").strip() or filename
    authors_raw = data.get("authors") or []
    authors = [str(a).strip() for a in authors_raw if isinstance(a, (str, int, float)) and str(a).strip()]
    date = data.get("date")
    date = str(date).strip() if date else None
    summary = str(data.get("summary") or "").strip()
    return {"title": title, "authors": authors, "date": date, "summary": summary}


def build_signature(meta: dict) -> str:
    parts: List[str] = []
    title = (meta.get("title") or "").strip()
    if title:
        parts.append(title + ".")
    authors = meta.get("authors") or []
    if authors:
        names = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        parts.append("By " + names + ".")
    date = meta.get("date")
    if date:
        parts.append(str(date) + ".")
    summary = (meta.get("summary") or "").strip()
    if summary:
        parts.append(summary if summary.endswith(".") else summary + ".")
    return " ".join(parts).strip()


def build_embed_text(signature: str, chunk: "Chunk") -> str:
    lines: List[str] = []
    if signature:
        lines.append(signature)
    headings = getattr(chunk, "headings", None) or []
    if headings:
        lines.append("Section: " + " > ".join(headings) + ".")
    lines.append(chunk.text)
    return "\n\n".join(lines)


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
