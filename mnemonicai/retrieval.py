"""Reconstructive, cue-driven retrieval.

Ranks candidate memories by a blend of cue similarity, recency, current
strength, and spreading activation across associative links from whatever is
currently active in working memory.
"""
from __future__ import annotations

from typing import List, Sequence

from .config import Config
from .dynamics import current_strength, recency
from .memory_item import MemoryItem
from .stores import LongTermStore, WorkingMemory
from .vectors import cosine


class RetrievalEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def retrieve(self, *args, **kwargs) -> List[MemoryItem]:
        return [m for _, m in self.retrieve_scored(*args, **kwargs)]

    def retrieve_scored(
        self,
        cue_embedding: Sequence[float],
        stores: List[LongTermStore],
        working: WorkingMemory,
        now: float,
        active_items: List[MemoryItem],
        k: int = 6,
        include_working: bool = True,
    ):
        """Like retrieve, but returns (score, item) pairs for reporting."""
        cfg = self.cfg
        candidates: List[MemoryItem] = []
        for store in stores:
            candidates.extend(store.all())
        if include_working:
            candidates.extend(working.snapshot())

        scored = []
        for m in candidates:
            sim = cosine(cue_embedding, m.embedding)
            rec = recency(m, now, cfg)
            stg = current_strength(m, now, cfg)
            spread = 0.0
            for a in active_items:
                if a.id == m.id:
                    continue
                spread += a.activation * a.links.get(m.id, 0.0)
            score = (cfg.w_sim * sim
                     + cfg.w_rec * rec
                     + cfg.w_str * stg
                     + cfg.w_assoc * spread)
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate: recall shouldn't surface the same memory twice just
        # because it exists as both an episode and a distilled fact.
        results = []
        for score, m in scored:
            if score < cfg.retrieval_floor:
                continue
            if any(cosine(m.embedding, r.embedding) >= cfg.dup_threshold for _, r in results):
                continue
            results.append((score, m))
            if len(results) >= k:
                break
        return results
