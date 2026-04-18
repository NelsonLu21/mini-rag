"""In-memory hybrid index: dense embeddings (cosine) + BM25 keyword.

Persisted to a single pickle on disk. No third-party vector DB.
"""
from __future__ import annotations
import math, os, pickle, re, threading
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from .ingest import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(s)]


@dataclass
class Store:
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: list = field(default_factory=list)  # list[np.ndarray]
    df: Counter = field(default_factory=Counter)    # doc-frequency for BM25
    tf: List[Counter] = field(default_factory=list)  # per-chunk term freq
    lens: List[int] = field(default_factory=list)
    avgdl: float = 0.0

    def add(self, chunks: List[Chunk], embs: List[List[float]]):
        for c, e in zip(chunks, embs):
            self.chunks.append(c)
            self.embeddings.append(_norm(np.array(e, dtype=np.float32)))
            toks = tokenize(c.text)
            tf = Counter(toks)
            self.tf.append(tf)
            self.lens.append(len(toks))
            for term in tf:
                self.df[term] += 1
        self.avgdl = sum(self.lens) / max(1, len(self.lens))

    def remove_doc(self, doc: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc != doc]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        self.tf = [self.tf[i] for i in keep]
        self.lens = [self.lens[i] for i in keep]
        self.df = Counter()
        for tf in self.tf:
            for term in tf:
                self.df[term] += 1
        self.avgdl = sum(self.lens) / max(1, len(self.lens)) if self.lens else 0.0
        return removed

    # ---- search ----
    def dense(self, qvec: np.ndarray, k: int) -> List[tuple[int, float]]:
        if not self.embeddings:
            return []
        M = np.vstack(self.embeddings)
        scores = M @ _norm(qvec)
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def bm25(self, query: str, k: int, k1: float = 1.5, b: float = 0.75) -> List[tuple[int, float]]:
        if not self.chunks:
            return []
        q = tokenize(query)
        N = len(self.chunks)
        scores = np.zeros(N, dtype=np.float32)
        for term in q:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self.lens[i]
                denom = f + k1 * (1 - b + b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (k1 + 1)) / denom
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


# --- persistence ----------------------------------------------------------
_LOCK = threading.Lock()
_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "store.pkl")


def load() -> Store:
    if os.path.exists(_PATH):
        with open(_PATH, "rb") as f:
            return pickle.load(f)
    return Store()


def save(store: Store) -> None:
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with _LOCK, open(_PATH, "wb") as f:
        pickle.dump(store, f)
