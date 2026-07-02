"""The memory bridge — wraps every chat request with brain-like memory.

Per request: perceive the user's message (encoding + salience gate), retrieve
relevant long-term memories and inject them into the prompt, let the model
generate, then perceive the reply. Every few turns it runs a sleep pass
(consolidate short-term → long-term) and, optionally, bakes memories into the
LoRA weights via the SleepTrainer.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .memory_item import MemoryItem


class MemoryChat:
    def __init__(self, memory, backend, bus, trainer, cfg) -> None:
        self.mem = memory
        self.backend = backend
        self.bus = bus
        self.trainer = trainer
        self.cfg = cfg
        self.turns = 0
        # route engine events onto the bus for the live monitor
        self.mem.on_event = bus.publish

    # ---- build the augmented prompt ----
    def build_messages(self, request_messages: List[Dict[str, str]]
                       ) -> Tuple[List[Dict[str, str]], List[MemoryItem]]:
        user_msgs = [m for m in request_messages if m.get("role") == "user"]
        last = user_msgs[-1]["content"] if user_msgs else ""
        if last:
            self.mem.perceive(last, source="user", importance=self.cfg.perceive_importance)
        recalled = self.mem.retrieve(last, k=self.cfg.recall_k) if last else []

        msgs = [dict(m) for m in request_messages]
        block = self._format_memories(recalled)
        if block:
            sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
            if sys_idx is None:
                msgs.insert(0, {"role": "system", "content": block})
            else:
                msgs[sys_idx]["content"] = msgs[sys_idx]["content"].rstrip() + "\n\n" + block
        self.bus.publish({"type": "chat", "role": "user", "preview": last[:120]})
        return msgs, recalled

    @staticmethod
    def _format_memories(memories: List[MemoryItem]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories (recalled from long-term store; use if helpful, "
                 "do not fabricate):"]
        for m in memories:
            lines.append(f"- ({m.kind.value}) {m.content}")
        return "\n".join(lines)

    # ---- close the loop after generation ----
    def after_reply(self, reply: str) -> None:
        if reply:
            self.mem.perceive(reply, source="self", importance=0.4)
        self.mem.tick(dt=1800.0)  # ~30 min of simulated time per exchange
        self.bus.publish({"type": "chat", "role": "assistant", "preview": reply[:120]})
        self.turns += 1
        if self.cfg.sleep_every_n_turns and self.turns % self.cfg.sleep_every_n_turns == 0:
            self.mem.sleep()  # emits a "consolidate" event
            if self.cfg.train_on_sleep:
                self.trainer.consolidate_to_weights()  # emits "train" events
        self.publish_state()

    def state_dict(self, max_memories: int = 150) -> dict:
        s = self.mem.stats()
        now = self.mem.now()
        longterm = self.mem._all_longterm()
        total_links = sum(len(m.links) for m in longterm)
        # each dot is a memory: id, kind, current strength, times recalled
        from .dynamics import current_strength
        mems = sorted(longterm, key=lambda m: current_strength(m, now, self.mem.cfg),
                      reverse=True)[:max_memories]
        memories = [{"id": m.id, "kind": m.kind.value,
                     "strength": round(current_strength(m, now, self.mem.cfg), 3),
                     "access": m.access_count, "text": m.short(70),
                     "pinned": bool(m.metadata.get("pinned"))} for m in mems]
        # Hebbian link graph among the reported memories (for the graph view)
        id_set = {m.id for m in mems}
        pairs = {}
        for m in mems:
            for oid, wt in m.links.items():
                if oid in id_set:
                    key = (m.id, oid) if m.id < oid else (oid, m.id)
                    if wt > pairs.get(key, 0.0):
                        pairs[key] = wt
        edges = [{"a": a, "b": b, "w": round(w, 3)}
                 for (a, b), w in sorted(pairs.items(), key=lambda kv: -kv[1])[:400]]
        return {
            "type": "state",
            "working": [{"id": w.id, "text": w.short(60), "activation": round(w.activation, 3)}
                        for w in self.mem.working.snapshot()],
            "memories": memories,
            "edges": edges,
            "counts": {"episodic": s["episodic"], "semantic": s["semantic"],
                       "procedural": s["procedural"]},
            "avg_strength": s["avg_strength_episodic"],
            "recalls": getattr(self.mem, "recalls", 0),
            "forgotten": getattr(self.mem, "forgotten", 0),
            "links": total_links // 2,
            "adapter_version": getattr(self.backend, "adapter_version", 0),
        }

    # ---- admin actions used by the monitor control bar ----
    def admin_perceive(self, text: str, importance: float = None) -> dict:
        imp = self.cfg.perceive_importance if importance is None else float(importance)
        item = self.mem.perceive(text, source="user", importance=imp)
        self.publish_state()
        return {"ok": True, "admitted": item is not None}

    def admin_recall(self, cue: str) -> dict:
        res = self.mem.retrieve(cue, k=self.cfg.recall_k) if cue else []
        self.publish_state()
        return {"ok": True, "hits": [{"id": m.id, "kind": m.kind.value, "text": m.short(90)}
                                     for m in res]}

    def admin_sleep(self) -> dict:
        rep = self.mem.sleep()
        self.publish_state()
        return {"ok": True, "report": rep}

    def admin_train(self) -> dict:
        res = self.trainer.consolidate_to_weights()
        self.publish_state()
        return {"ok": True, "result": res}

    def admin_reset(self) -> dict:
        self.mem.episodic.items.clear()
        self.mem.semantic.items.clear()
        self.mem.procedural.items.clear()
        self.mem.working.items.clear()
        self.mem._staging = []
        self.mem.recalls = 0
        self.mem.forgotten = 0
        self.turns = 0
        self.publish_state()
        return {"ok": True}

    def admin_delete(self, mem_id: str) -> dict:
        removed = False
        for store in (self.mem.episodic, self.mem.semantic, self.mem.procedural):
            if store.get(mem_id) is not None:
                store.remove(mem_id)
                removed = True
        self.publish_state()
        return {"ok": removed}

    def admin_pin(self, mem_id: str) -> dict:
        pinned = None
        for store in (self.mem.episodic, self.mem.semantic, self.mem.procedural):
            m = store.get(mem_id)
            if m is not None:
                m.metadata["pinned"] = not m.metadata.get("pinned")
                pinned = m.metadata["pinned"]
                break
        self.publish_state()
        return {"ok": pinned is not None, "pinned": pinned}

    def admin_memories(self, query: str = "") -> dict:
        now = self.mem.now()
        from .dynamics import current_strength
        out = []
        for m in self.mem._all_longterm():
            if query and query.lower() not in m.content.lower():
                continue
            out.append({"id": m.id, "kind": m.kind.value, "text": m.content,
                        "strength": round(current_strength(m, now, self.mem.cfg), 3),
                        "access": m.access_count, "pinned": bool(m.metadata.get("pinned"))})
        out.sort(key=lambda x: x["strength"], reverse=True)
        return {"memories": out[:300]}

    def publish_state(self) -> None:
        self.bus.publish(self.state_dict())

    # ---- the two entry points the server uses ----
    def complete(self, request_messages: List[Dict[str, str]], max_new_tokens=None) -> str:
        msgs, _ = self.build_messages(request_messages)
        reply = self.backend.generate(msgs, max_new_tokens=max_new_tokens)
        self.after_reply(reply)
        return reply

    def stream(self, request_messages: List[Dict[str, str]], max_new_tokens=None):
        msgs, _ = self.build_messages(request_messages)
        chunks = []
        for delta in self.backend.generate_stream(msgs, max_new_tokens=max_new_tokens):
            chunks.append(delta)
            yield delta
        self.after_reply("".join(chunks))
