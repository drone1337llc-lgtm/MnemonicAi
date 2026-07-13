"""Per-tenant in-process memory — the real isolation mechanism.

The shared pod runs ONE raw LLM engine (fast, GPU-bound), but each tenant
gets their own BrainMemory loaded from their own isolated memory.db. This
manager holds an LRU set of active tenant brains, persisting + evicting the
cold ones. A tenant's chat only ever perceives/recalls against their own
brain, so no cross-tenant leakage is possible — Aerith literally cannot see
another user's memories because they aren't loaded in that request's context.

Sits in the gateway. Uses the shipped mnemonicai package for the brain.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict

# the mnemonicai package ships alongside cloud/ on the pod
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mnemonicai.memory_system import BrainMemory      # noqa: E402
from mnemonicai.config import Config                  # noqa: E402


class TenantBrains:
    def __init__(self, max_active: int = 32, embedder=None,
                 persist_every_s: int = 120) -> None:
        self.max_active = max_active
        self.embedder = embedder
        self._brains: "OrderedDict[str, BrainMemory]" = OrderedDict()
        self._paths: dict[str, str] = {}
        self._lock = threading.RLock()
        threading.Thread(target=self._autosave_loop,
                         args=(persist_every_s,), daemon=True).start()

    def _load(self, tenant_id: str, db_path: str) -> BrainMemory:
        mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=time.time,
                          embedder=self.embedder)
        if os.path.isfile(db_path):
            try:
                mem.load(db_path)
            except Exception:
                pass
        return mem

    def get(self, tenant_id: str, db_path: str) -> BrainMemory:
        with self._lock:
            if tenant_id in self._brains:
                self._brains.move_to_end(tenant_id)
                return self._brains[tenant_id]
            # evict coldest if at capacity (persist it first)
            while len(self._brains) >= self.max_active:
                old_id, old_mem = self._brains.popitem(last=False)
                self._persist(old_id, old_mem)
            mem = self._load(tenant_id, db_path)
            self._brains[tenant_id] = mem
            self._paths[tenant_id] = db_path
            return mem

    def _persist(self, tenant_id: str, mem: BrainMemory) -> None:
        path = self._paths.get(tenant_id)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mem.save(path)
        except Exception:
            pass

    def persist(self, tenant_id: str) -> None:
        with self._lock:
            m = self._brains.get(tenant_id)
            if m:
                self._persist(tenant_id, m)

    def persist_all(self) -> None:
        with self._lock:
            for tid, m in self._brains.items():
                self._persist(tid, m)

    def _autosave_loop(self, every_s: int) -> None:
        while True:
            time.sleep(every_s)
            self.persist_all()

    # ---- the tenant-scoped chat cycle ----
    def augment(self, tenant_id: str, db_path: str, messages: list,
                recall_k: int = 6) -> list:
        """Perceive the user's message into THIS tenant's brain and inject
        THEIR recalled memories. Returns the augmented message list."""
        mem = self.get(tenant_id, db_path)
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        if user:
            mem.perceive(user, source="user", importance=0.6)
        recalled = mem.retrieve(user, k=recall_k) if user else []
        if not recalled:
            return messages
        block = ("Relevant memories (recalled from this user's private store):\n"
                 + "\n".join(f"- {m.content}" for m in recalled))
        out = [dict(m) for m in messages]
        sys_i = next((i for i, m in enumerate(out)
                      if m.get("role") == "system"), None)
        if sys_i is None:
            out.insert(0, {"role": "system", "content": block})
        else:
            out[sys_i]["content"] = out[sys_i]["content"].rstrip() + "\n\n" + block
        return out

    def remember_reply(self, tenant_id: str, db_path: str, reply: str) -> None:
        if not reply:
            return
        mem = self.get(tenant_id, db_path)
        mem.perceive(reply, source="self", importance=0.4)
        mem.tick(dt=1800.0)

    def stats(self, tenant_id: str, db_path: str) -> dict:
        return self.get(tenant_id, db_path).stats()
