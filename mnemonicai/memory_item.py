"""The universal memory trace.

Every memory in every store -- sensory, working, episodic, semantic,
procedural -- is a `MemoryItem`. A single uniform schema is what lets decay,
retrieval, reinforcement, and associative linking work identically across all
memory types.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class MemoryKind(str, Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryItem:
    content: str
    kind: MemoryKind = MemoryKind.WORKING
    embedding: List[float] = field(default_factory=list)

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = 0.0
    last_access_at: float = 0.0
    access_count: int = 0

    # `salience` = how important this was at encoding (fixed, never decays).
    # `strength` = how well it is currently retained AT last_access_at (decays
    #              on-demand between accesses; see dynamics.current_strength).
    # `stability` = tau multiplier; grows with reinforcement (spacing effect).
    # `activation` = transient working-memory energy + spreading-activation.
    salience: float = 0.5
    strength: float = 1.0
    stability: float = 1.0
    activation: float = 0.0

    links: Dict[str, float] = field(default_factory=dict)   # other_id -> weight
    summary_of: List[str] = field(default_factory=list)     # ids this gist covers
    source: str = ""                                        # provenance
    metadata: dict = field(default_factory=dict)

    # ---- (de)serialization helpers used by the SQLite backend ----
    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        d = dict(d)
        d["kind"] = MemoryKind(d.get("kind", "working"))
        return cls(**d)

    def short(self, n: int = 60) -> str:
        c = self.content.replace("\n", " ")
        return c if len(c) <= n else c[: n - 1] + "…"
