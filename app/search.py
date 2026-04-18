"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion.

Why RRF: scale-free combiner — dense cosine and BM25 produce scores on totally
different scales, so weighted sums need calibration. RRF only uses ranks, so it
combines them cleanly without tuning.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from .store import Store
from .ingest import Chunk
from . import mistral


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused RRF score
    dense: float          # raw cosine in [-1, 1]
    bm25: float


def hybrid(store: Store, query: str, k: int = 8, k_rrf: int = 60) -> List[Hit]:
    if not store.chunks:
        return []
    qvec = np.array(mistral.embed([query])[0], dtype=np.float32)
    dense = store.dense(qvec, k=max(k * 4, 20))
    sparse = store.bm25(query, k=max(k * 4, 20))

    fused: dict[int, float] = {}
    for rank, (i, _) in enumerate(dense):
        fused[i] = fused.get(i, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, (i, _) in enumerate(sparse):
        fused[i] = fused.get(i, 0.0) + 1.0 / (k_rrf + rank + 1)

    dense_map = dict(dense)
    sparse_map = dict(sparse)

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return [
        Hit(
            chunk=store.chunks[i],
            score=s,
            dense=dense_map.get(i, 0.0),
            bm25=sparse_map.get(i, 0.0),
        )
        for i, s in ranked
    ]
