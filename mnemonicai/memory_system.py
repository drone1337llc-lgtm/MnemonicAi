"""BrainMemory -- the public facade tying the whole pipeline together.

Typical use inside an agent loop:

    mem = BrainMemory()                       # offline defaults
    mem.set_goal("plan a trip to Japan")
    mem.perceive("The user is vegetarian.", source="user")
    recalled = mem.retrieve("what does the user eat?")
    mem.tick()                                # advance time / decay working set
    mem.sleep()                               # consolidate + prune
    mem.save("agent_memory.db")
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from .config import Config
from .consolidation import ConsolidationEngine
from .dynamics import current_strength, reinforce
from .embeddings import HashingEmbedder
from .llm import HeuristicLLM
from .memory_item import MemoryItem, MemoryKind
from .retrieval import RetrievalEngine
from .salience import SalienceScorer
from .stores import LongTermStore, SemanticStore, SensoryBuffer, WorkingMemory
from . import persistence


class BrainMemory:
    def __init__(self, config: Optional[Config] = None, embedder=None, llm=None,
                 clock: Optional[Callable[[], float]] = None,
                 on_event: Optional[Callable[[dict], None]] = None) -> None:
        self.cfg = config or Config()
        self.embedder = embedder or HashingEmbedder(self.cfg.embedding_dim)
        self.llm = llm or HeuristicLLM()
        self.clock = clock or time.time

        self.sensory = SensoryBuffer()
        self.working = WorkingMemory(self.cfg)
        self.episodic = LongTermStore(MemoryKind.EPISODIC)
        self.semantic = SemanticStore()
        self.procedural = LongTermStore(MemoryKind.PROCEDURAL)

        self.salience = SalienceScorer(self.cfg)
        self.retriever = RetrievalEngine(self.cfg)
        self.consolidator = ConsolidationEngine(self.cfg, self.embedder, self.llm)

        self._staging: List[MemoryItem] = []   # evicted WM items awaiting sleep
        self._tick_count = 0
        self.goal_embedding: Optional[List[float]] = None
        self.archive: List[MemoryItem] = []
        self.on_event = on_event               # callable(dict) for live monitoring
        self.recalls = 0
        self.forgotten = 0

    # ---- clock -------------------------------------------------------
    def now(self) -> float:
        return self.clock()

    def _emit(self, etype: str, **fields) -> None:
        cb = self.on_event
        if cb is not None:
            try:
                cb(dict(type=etype, **fields))
            except Exception:
                pass  # monitoring must never break the engine

    def set_goal(self, text: str) -> None:
        self.goal_embedding = self.embedder.embed([text])[0]

    # ---- perceive (encode) ------------------------------------------
    def perceive(self, content: str, source: str = "", importance: float = 0.0,
                 emotion: Optional[float] = None, metadata: Optional[dict] = None
                 ) -> Optional[MemoryItem]:
        """Take in one observation. Returns the working-memory item if it passed
        the salience gate, or None if it was filtered out (forgotten at intake)."""
        now = self.now()
        emb = self.embedder.embed([content])[0]
        item = MemoryItem(content=content, kind=MemoryKind.SENSORY, embedding=emb,
                          created_at=now, last_access_at=now,
                          source=source, metadata=metadata or {})
        self.sensory.add(item)

        existing = ([m.embedding for m in self._all_longterm()]
                    + [w.embedding for w in self.working.snapshot()])
        sal, brk = self.salience.score_detailed(
            emb, content, existing, importance=importance,
            goal_embedding=self.goal_embedding, emotion=emotion)
        item.salience = sal
        if metadata:
            item.metadata.update(metadata)

        admitted = sal >= self.cfg.salience_threshold
        self._emit("gate", id=item.id, text=item.short(120), source=source,
                   salience=round(sal, 3), admitted=admitted, **brk)
        if admitted:
            evicted = self.working.add(item)
            self._emit("working_add", id=item.id, text=item.short(120),
                       salience=round(sal, 3), activation=round(item.activation, 3))
            for ev in evicted:
                self._emit("working_evict", id=ev.id, text=ev.short(80))
            self._stage(evicted)
            return item
        self.forgotten += 1
        return None  # below the gate: immediately forgotten

    def rehearse(self, item_id: str) -> None:
        """Actively keep an item alive in working memory (maintenance rehearsal)."""
        self.working.rehearse(item_id)

    # ---- retrieve (reconstructive recall) ---------------------------
    def retrieve(self, cue: str, k: int = 6, include_working: bool = True
                 ) -> List[MemoryItem]:
        now = self.now()
        emb = self.embedder.embed([cue])[0]
        active = self.working.snapshot()
        scored = self.retriever.retrieve_scored(
            emb, [self.episodic, self.semantic, self.procedural],
            self.working, now, active, k=k, include_working=include_working)
        results = [m for _, m in scored]
        self._emit("retrieve", cue=cue[:120],
                   hits=[{"id": m.id, "kind": m.kind.value, "score": round(sc, 3),
                          "text": m.short(80)} for sc, m in scored])
        for sc, m in scored:
            reinforce(m, now, self.cfg)          # spacing effect
            self._emit("reinforce", id=m.id, kind=m.kind.value,
                       strength=round(m.strength, 3), stability=round(m.stability, 3))
        if scored:
            self.recalls += 1
        self._hebbian(active + results)          # co-activation wires together
        return results

    # ---- time + sleep -----------------------------------------------
    def tick(self, dt: Optional[float] = None) -> None:
        """Advance one step. If a manual clock is used, `dt` advances it."""
        if dt is not None and hasattr(self.clock, "advance"):
            self.clock.advance(dt)  # type: ignore[attr-defined]
        self.working.tick()
        self._stage(self.working.pop_below(self.cfg.wm_stage_threshold))
        self._tick_count += 1
        if self.cfg.sleep_every_n_ticks and self._tick_count % self.cfg.sleep_every_n_ticks == 0:
            self.sleep()

    def sleep(self) -> dict:
        """Offline consolidation pass + pruning. The short-term -> long-term step."""
        now = self.now()
        candidates = self._staging + self.working.snapshot()
        report = self.consolidator.sleep(
            candidates, self.episodic, self.semantic, self.procedural, now)
        self._staging = []
        report["pruned"] = self._prune(now)
        self._emit("consolidate",
                   episodic=report.get("episodic", 0), semantic=report.get("semantic", 0),
                   procedural=report.get("procedural", 0), gists=report.get("gists", 0),
                   links=report.get("links", 0), pruned=report.get("pruned", 0))
        return report

    # ---- introspection ----------------------------------------------
    def stats(self) -> dict:
        now = self.now()
        def avg_strength(store):
            items = store.all()
            if not items:
                return 0.0
            return round(sum(current_strength(i, now, self.cfg) for i in items) / len(items), 3)
        return {
            "working": len(self.working.items),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "procedural": len(self.procedural),
            "avg_strength_episodic": avg_strength(self.episodic),
            "avg_strength_semantic": avg_strength(self.semantic),
        }

    def recall_strength(self, item: MemoryItem) -> float:
        return current_strength(item, self.now(), self.cfg)

    # ---- persistence -------------------------------------------------
    def save(self, path: str) -> None:
        persistence.save(self, path)

    def load(self, path: str) -> None:
        persistence.load(self, path)

    # ---- internals ---------------------------------------------------
    def _all_longterm(self) -> List[MemoryItem]:
        return self.episodic.all() + self.semantic.all() + self.procedural.all()

    def _stage(self, items: List[MemoryItem]) -> None:
        for it in items:
            self._staging.append(it)

    def _hebbian(self, items: List[MemoryItem]) -> None:
        cfg = self.cfg
        uniq = {it.id: it for it in items}
        vals = list(uniq.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                w = min(cfg.link_cap, a.links.get(b.id, 0.0) + cfg.link_increment)
                a.links[b.id] = w
                b.links[a.id] = w

    def _prune(self, now: float) -> int:
        pruned = 0
        for store in (self.episodic, self.semantic, self.procedural):
            for m in store.all():
                if m.metadata.get("pinned"):
                    continue  # pinned memories are never forgotten
                if (current_strength(m, now, self.cfg) < self.cfg.strength_floor
                        and m.access_count < self.cfg.keep_access_count):
                    if self.cfg.archive_on_evict:
                        self.archive.append(m)
                    store.remove(m.id)
                    pruned += 1
        self.forgotten += pruned
        return pruned


class ManualClock:
    """A controllable clock for demos and tests. Call `advance(seconds)`."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
