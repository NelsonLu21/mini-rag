"""PDF text extraction and chunking.

Chunking considerations:
- Fixed-size token/char chunks miss semantic boundaries; pure sentence chunks vary
  wildly in length and hurt embedding quality. We split on paragraph boundaries
  first, then pack paragraphs into ~target_chars windows.
- Overlap (sliding window) preserves context that straddles a chunk boundary so
  retrieval doesn't lose answers split across chunks.
- We keep page numbers per chunk for citation.
- We strip excessive whitespace; PDFs often produce ragged line breaks mid-sentence
  which we collapse so embeddings see fluent text.
- For very long paragraphs we hard-split on character count to avoid one giant chunk.
- We do NOT do layout-aware extraction (tables, columns) — out of scope; pypdf gives
  reading-order text which is good enough for a baseline.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader


@dataclass
class Chunk:
    doc: str
    page: int
    text: str
    idx: int


def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    # join hyphenated line breaks: "exam-\nple" -> "example"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # collapse single newlines inside paragraphs but keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pages(path: str) -> List[tuple[int, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i + 1, _clean(txt)))
    return out


def chunk_pages(
    pages: List[tuple[int, str]],
    doc: str,
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for page, text in pages:
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        buf = ""
        for p in paragraphs:
            # hard split paragraphs that are themselves too long
            for piece in _hard_split(p, target_chars):
                if len(buf) + len(piece) + 1 <= target_chars or not buf:
                    buf = (buf + " " + piece).strip()
                else:
                    chunks.append(Chunk(doc, page, buf, idx)); idx += 1
                    tail = buf[-overlap_chars:] if overlap_chars else ""
                    buf = (tail + " " + piece).strip()
        if buf:
            chunks.append(Chunk(doc, page, buf, idx)); idx += 1
    return chunks


def _hard_split(text: str, n: int) -> List[str]:
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(0, len(text), n)]
