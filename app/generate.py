"""Answer generation: intent-templated prompts + post-hoc evidence check."""
from __future__ import annotations
import re
from typing import List, Tuple
from . import mistral
from .search import Hit

_CITE_RX = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _filter_used(text: str, n_context: int) -> Tuple[str, List[int]]:
    order: List[int] = []
    seen = set()
    for m in _CITE_RX.finditer(text):
        for s in m.group(1).split(","):
            try:
                n = int(s.strip())
            except ValueError:
                continue
            if 1 <= n <= n_context and n not in seen:
                seen.add(n)
                order.append(n)
    if not order:
        return text, []
    mapping = {old: new for new, old in enumerate(order, 1)}

    def sub(m):
        parts = []
        for s in m.group(1).split(","):
            try:
                n = int(s.strip())
            except ValueError:
                continue
            if n in mapping:
                parts.append(str(mapping[n]))
        return "[" + ", ".join(parts) + "]" if parts else ""

    return _CITE_RX.sub(sub, text), order

SIM_THRESHOLD = 0.55  # min top-1 cosine to attempt an answer

_BASE = (
    "You are a careful assistant answering questions strictly from the provided "
    "context excerpts. Cite sources inline as [n] matching the excerpt numbers. "
    "If the context does not contain the answer, reply exactly: "
    '"I don\'t have enough information in the knowledge base to answer that."'
)

_STYLES = {
    "qa": "Answer in 1-3 short paragraphs.",
    "list": "Answer as a markdown bullet list. Each bullet ends with its [n] citation.",
    "table": "Answer as a markdown table with a header row. Include a Source column with [n] citations.",
}


def _format_context(hits: List[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits):
        loc = f"{h.chunk.doc} p.{h.chunk.page}"
        headings = getattr(h.chunk, "headings", None) or []
        if headings:
            loc += " > " + " > ".join(headings)
        parts.append(f"[{i+1}] ({loc})\n{h.chunk.text}")
    return "\n\n".join(parts)


def answer(query: str, hits: List[Hit], intent: str) -> dict:
    if not hits or hits[0].dense < SIM_THRESHOLD:
        return {
            "answer": "Insufficient evidence in the knowledge base to answer that confidently.",
            "citations": [],
            "evidence_check": {"checked": False, "unsupported": []},
        }

    style = _STYLES.get(intent, _STYLES["qa"])
    prompt = (
        f"{_BASE}\n\n{style}\n\nContext:\n{_format_context(hits)}\n\n"
        f"Question: {query}"
    )
    text = mistral.chat(
        [{"role": "system", "content": _BASE + " " + style},
         {"role": "user", "content": prompt}],
        temperature=0.1,
    )

    renumbered, used_order = _filter_used(text, len(hits))
    used_hits = [hits[i - 1] for i in used_order]
    unsupported = evidence_check(renumbered, used_hits or hits)
    return {
        "answer": renumbered,
        "citations": [
            {"n": i + 1, "doc": h.chunk.doc, "page": h.chunk.page,
             "headings": getattr(h.chunk, "headings", None) or [],
             "similarity": round(h.dense, 4),
             "text": h.chunk.text}
            for i, h in enumerate(used_hits)
        ],
        "evidence_check": {"checked": True, "unsupported": unsupported},
    }


def evidence_check(text: str, hits: List[Hit]) -> List[str]:
    """Flag answer sentences that don't have lexical or semantic support.

    Cheap two-stage filter:
      1. Token overlap with any context chunk ≥ 0.25 → supported.
      2. Otherwise embed the sentence and check max cosine with chunks ≥ 0.6.
    """
    import re, numpy as np
    from .store import tokenize

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    if not sentences:
        return []

    ctx_tokens = [set(tokenize(h.chunk.text)) for h in hits]
    weak: List[str] = []
    for s in sentences:
        st = set(tokenize(s))
        if not st:
            continue
        overlap = max((len(st & ct) / len(st) for ct in ctx_tokens), default=0)
        if overlap < 0.25:
            weak.append(s)

    if not weak:
        return []

    try:
        chunk_embs = np.array(mistral.embed([h.chunk.text for h in hits]), dtype=np.float32)
        chunk_embs /= (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
        sent_embs = np.array(mistral.embed(weak), dtype=np.float32)
        sent_embs /= (np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-9)
        sims = sent_embs @ chunk_embs.T
        return [s for s, sim in zip(weak, sims.max(axis=1)) if sim < 0.6]
    except Exception:
        return weak
