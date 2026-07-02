"""Memory stores: the sensory buffer, working memory, and the long-term stores.

All long-term stores share one implementation (`LongTermStore`) because decay,
retrieval, and linking treat every trace uniformly. The semantic store adds a
lightweight keyword concept-index on top for graph-style lookups.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config
from .memory_item import MemoryItem, MemoryKind
from .vectors import cosine


class SensoryBuffer:
    """A short ring buffer of the most recent raw inputs. Almost nothing survives."""

    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def drain(self) -> List[MemoryItem]:
        items = self.items
        self.items = []
        return items


class WorkingMemory:
    """Capacity-bounded, decaying active set. The agent's short-term workspace."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.items: Dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> List[MemoryItem]:
        """Insert an item; return any items evicted to keep within capacity."""
        item.kind = MemoryKind.WORKING
        item.activation = 1.0
        self.items[item.id] = item
        evicted: List[MemoryItem] = []
        while len(self.items) > self.cfg.working_capacity:
            victim_id = min(self.items, key=lambda k: self.items[k].activation)
            evicted.append(self.items.pop(victim_id))
        return evicted

    def rehearse(self, item_id: str) -> None:
        it = self.items.get(item_id)
        if it is not None:
            it.activation = min(1.0, it.activation + self.cfg.rehearse_boost)

    def tick(self) -> None:
        for it in self.items.values():
            it.activation *= (1.0 - self.cfg.activation_decay)

    def snapshot(self) -> List[MemoryItem]:
        return list(self.items.values())

    def pop_below(self, threshold: float) -> List[MemoryItem]:
        """Remove and return items whose activation has fallen below threshold."""
        gone = [it for it in self.items.values() if it.activation < threshold]
        for it in gone:
            self.items.pop(it.id, None)
        return gone


class LongTermStore:
    """A durable collection of memory traces (episodic / semantic / procedural)."""

    def __init__(self, kind: MemoryKind) -> None:
        self.kind = kind
        self.items: Dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        item.kind = self.kind
        self.items[item.id] = item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self.items.get(item_id)

    def remove(self, item_id: str) -> None:
        self.items.pop(item_id, None)

    def all(self) -> List[MemoryItem]:
        return list(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    def most_similar(self, embedding: Sequence[float], k: int = 1
                     ) -> List[Tuple[float, MemoryItem]]:
        scored = [(cosine(embedding, it.embedding), it) for it in self.items.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


class SemanticStore(LongTermStore):
    """Long-term facts plus a keyword concept-index for graph-style lookups."""

    def __init__(self) -> None:
        super().__init__(MemoryKind.SEMANTIC)
        self.concept_index: Dict[str, List[str]] = {}

    def add(self, item: MemoryItem) -> None:
        super().add(item)
        for concept in _concepts(item.content):
            self.concept_index.setdefault(concept, [])
            if item.id not in self.concept_index[concept]:
                self.concept_index[concept].append(item.id)

    def by_concept(self, concept: str) -> List[MemoryItem]:
        ids = self.concept_index.get(concept.lower(), [])
        return [self.items[i] for i in ids if i in self.items]


_STOP = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
         "in", "on", "at", "for", "with", "as", "by", "it", "this", "that",
         "has", "have", "had", "will", "can", "be", "user"}


def _concepts(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]
