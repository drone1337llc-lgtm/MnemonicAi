"""The consolidation ('sleep') engine.

Turns short-term traces into durable long-term memory:
  * promote items that clear the consolidation policy,
  * log every promoted item as an episode,
  * distill semantic facts (LLM or heuristic) with dedup + reconsolidation,
  * capture repeated actions as procedural recipes,
  * cluster near-duplicate episodes into gist summaries (schema formation),
  * wire Hebbian links between items consolidated together.
"""
from __future__ import annotations

import copy
from typing import Dict, List

from .config import Config
from .dynamics import current_strength, reinforce
from .memory_item import MemoryItem, MemoryKind
from .stores import LongTermStore, SemanticStore
from .vectors import blend, cosine


class ConsolidationEngine:
    def __init__(self, cfg: Config, embedder, llm) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.llm = llm

    def sleep(
        self,
        candidates: List[MemoryItem],
        episodic: LongTermStore,
        semantic: SemanticStore,
        procedural: LongTermStore,
        now: float,
    ) -> Dict[str, int]:
        cfg = self.cfg
        report = {"considered": len(candidates), "episodic": 0, "semantic": 0,
                  "procedural": 0, "gists": 0, "links": 0}

        promoted: List[MemoryItem] = []
        for it in candidates:
            passes = (current_strength(it, now, cfg) * it.salience >= cfg.theta_consolidate
                      or it.access_count >= cfg.k_rehearsals)
            if not passes:
                continue  # decays away without ever reaching long-term memory

            # --- Episodic: the event happened, so log it verbatim ---
            ep = copy.deepcopy(it)
            ep.kind = MemoryKind.EPISODIC
            ep.last_access_at = now
            episodic.add(ep)
            promoted.append(ep)
            report["episodic"] += 1

            # --- Semantic: distill atomic facts, dedup + reconsolidate ---
            for fact in self.llm.extract_facts(it.content):
                if self._upsert_fact(fact, semantic, now):
                    report["semantic"] += 1

            # --- Procedural: capture tool/action patterns as recipes ---
            if _looks_procedural(it):
                if self._upsert_procedure(it, procedural, now):
                    report["procedural"] += 1

        # --- Schema formation: cluster near-duplicate new episodes ---
        report["gists"] += self._form_gists(promoted, semantic, now)

        # --- Hebbian wiring: items consolidated together associate ---
        report["links"] += self._wire_links(promoted)

        return report

    # ------------------------------------------------------------------
    def _upsert_fact(self, fact: str, semantic: SemanticStore, now: float) -> bool:
        emb = self.embedder.embed([fact])[0]
        best = semantic.most_similar(emb, k=1)
        if best and best[0][0] >= self.cfg.dup_threshold:
            # Reconsolidation: strengthen and gently update the existing fact.
            existing = best[0][1]
            reinforce(existing, now, self.cfg)
            existing.embedding = blend(existing.embedding, emb, wa=0.7)
            existing.metadata.setdefault("revisions", 0)
            existing.metadata["revisions"] += 1
            return False
        item = MemoryItem(content=fact, kind=MemoryKind.SEMANTIC, embedding=emb,
                          created_at=now, last_access_at=now,
                          salience=0.7, strength=1.0, stability=1.2,
                          source="consolidation")
        semantic.add(item)
        return True

    def _upsert_procedure(self, src: MemoryItem, procedural: LongTermStore, now: float) -> bool:
        emb = src.embedding
        best = procedural.most_similar(emb, k=1)
        if best and best[0][0] >= self.cfg.dup_threshold:
            existing = best[0][1]
            reinforce(existing, now, self.cfg)  # practice strengthens a skill
            return False
        item = MemoryItem(content=src.content, kind=MemoryKind.PROCEDURAL, embedding=emb,
                          created_at=now, last_access_at=now,
                          salience=max(0.6, src.salience), strength=1.0, stability=1.5,
                          source=src.source or "procedure",
                          metadata={"skill": True})
        procedural.add(item)
        return True

    def _form_gists(self, promoted: List[MemoryItem], semantic: SemanticStore, now: float) -> int:
        gists = 0
        used = set()
        for i, a in enumerate(promoted):
            if a.id in used:
                continue
            cluster = [a]
            for b in promoted[i + 1:]:
                if b.id in used:
                    continue
                if cosine(a.embedding, b.embedding) >= self.cfg.dup_threshold:
                    cluster.append(b)
                    used.add(b.id)
            if len(cluster) >= 2:
                text = self.llm.summarize([c.content for c in cluster])
                if not text:
                    continue
                emb = self.embedder.embed([text])[0]
                gist = MemoryItem(content=text, kind=MemoryKind.SEMANTIC, embedding=emb,
                                  created_at=now, last_access_at=now,
                                  salience=0.8, strength=1.0, stability=1.5,
                                  summary_of=[c.id for c in cluster],
                                  source="gist")
                semantic.add(gist)
                gists += 1
        return gists

    def _wire_links(self, promoted: List[MemoryItem]) -> int:
        cfg = self.cfg
        n = 0
        for i in range(len(promoted)):
            for j in range(i + 1, len(promoted)):
                a, b = promoted[i], promoted[j]
                w = min(cfg.link_cap, a.links.get(b.id, 0.0) + cfg.link_increment)
                a.links[b.id] = w
                b.links[a.id] = w
                n += 1
        return n


def _looks_procedural(item: MemoryItem) -> bool:
    # Procedural memory is about *how to do* something -- skills and action
    # sequences -- not arbitrary facts that happened to come from a tool.
    if item.metadata.get("action") or item.metadata.get("skill"):
        return True
    if (item.source or "").lower().startswith("action"):
        return True
    low = item.content.lower()
    return low.startswith(("to ", "how to ", "step 1", "first,")) or " then " in low
