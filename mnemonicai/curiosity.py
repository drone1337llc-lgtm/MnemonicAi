"""Curiosity engine — intrinsic motivation for idle time.

When no one is talking to Aria, she explores her own memory: seeking
novelty, forming and testing hypotheses, walking associative links for gaps,
and inspecting her own statistics for weaknesses. Each exploration is scored
by INFORMATION GAIN — how far the new thought lands from everything already
remembered (1 - max cosine similarity in embedding space). High-gain insights
are stored as memories with salience proportional to the gain: the intrinsic
reward. Every cycle emits a "curiosity" event so the monitor shows her
wondering.
"""
from __future__ import annotations

import random
import threading
import time

from .vectors import cosine

_SYS = ("You are Aria reflecting privately during idle time. Think freely "
        "but answer with ONE concise, concrete insight (2-5 sentences). No "
        "preamble, no meta-commentary.")

MODES = ("novelty", "hypothesis", "graph", "introspect")


class CuriosityEngine(threading.Thread):
    def __init__(self, mem, backend, bus, cfg, chat) -> None:
        super().__init__(daemon=True, name="curiosity")
        self.mem, self.backend, self.bus, self.cfg, self.chat = \
            mem, backend, bus, cfg, chat
        self.recent: list[str] = []   # avoid re-chewing the same seeds

    # ---- helpers ----
    def _pool(self):
        return (self.mem.semantic.all() + self.mem.procedural.all()
                + self.mem.episodic.all())

    def _gain(self, text: str) -> float:
        try:
            v = self.mem.embedder.embed([text])[0]
        except Exception:
            return 0.0
        best = 0.0
        for m in self._pool():
            best = max(best, cosine(v, m.embedding or []))
        return 1.0 - best

    def _build_prompt(self, mode: str):
        pool = [m for m in self._pool() if m.id not in self.recent[-40:]]
        if len(pool) < 3:
            return None, []
        if mode == "novelty":
            a, b = random.sample(pool, 2)
            # prefer a semantically distant pair — that's where novelty lives
            for _ in range(6):
                c, d = random.sample(pool, 2)
                if cosine(c.embedding or [], d.embedding or []) < \
                   cosine(a.embedding or [], b.embedding or []):
                    a, b = c, d
            return (f"Two things you remember:\nA) {a.content}\nB) {b.content}\n"
                    "What unexpected connection, question, or idea arises from "
                    "holding these together?"), [a, b]
        if mode == "hypothesis":
            a, b = random.sample(pool, 2)
            return (f"From these memories:\n- {a.content}\n- {b.content}\n"
                    "Form one testable hypothesis, evaluate it against "
                    "everything you know, and state your conclusion plus what "
                    "evidence would change your mind."), [a, b]
        if mode == "graph":
            start = max(pool, key=lambda m: m.activation)
            chain, cur = [start], start
            for _ in range(2):
                nxt = None
                if cur.links:
                    lid = max(cur.links, key=cur.links.get)
                    nxt = next((m for m in pool if m.id == lid), None)
                if not nxt:
                    break
                chain.append(nxt)
                cur = nxt
            path = "\n-> ".join(m.content[:160] for m in chain)
            return (f"A path through your associative memory:\n{path}\n"
                    "What is the missing link or unexplored branch along this "
                    "path? Name it and say why it matters."), chain
        # introspect
        s = self.mem.stats()
        return (f"Your own memory statistics: {s}. Identify one concrete "
                "weakness in your current knowledge or behaviour and one "
                "specific, actionable way to improve it."), []

    # ---- one exploration ----
    def _cycle(self) -> None:
        mode = random.choice(MODES)
        print(f"[curiosity] cycle start mode={mode} pool={len(self._pool())}", flush=True)
        built = self._build_prompt(mode)
        if not built or not built[0]:
            print("[curiosity] no usable prompt this cycle", flush=True)
            return
        prompt, seeds = built
        print(f"[curiosity] generating for prompt: {prompt[:80]!r}", flush=True)
        reply = self.backend.generate(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": prompt}], max_new_tokens=300)
        reply = (reply or "").strip()
        print(f"[curiosity] reply len={len(reply)}", flush=True)
        if not reply:
            return
        gain = self._gain(reply)
        print(f"[curiosity] gain={gain:.3f}", flush=True)
        self.recent.extend(m.id for m in seeds)
        self.bus.publish({"type": "curiosity", "mode": mode,
                          "gain": round(gain, 3),
                          "preview": reply[:140]})
        floor = getattr(self.cfg, "curiosity_gain_floor", 0.35)
        if gain >= floor:
            # intrinsic reward: novel insights are remembered proportionally
            self.mem.perceive(f"(self-discovered while wondering — {mode}) {reply}",
                              source="curiosity",
                              importance=min(0.95, 0.45 + gain * 0.5))

    # ---- the idle loop ----
    def run(self) -> None:
        interval = max(120, getattr(self.cfg, "curiosity_every_s", 900))
        idle_gate = getattr(self.cfg, "curiosity_idle_s", 180)
        while True:
            time.sleep(interval)
            if not getattr(self.cfg, "curiosity_enabled", True):
                continue
            last = getattr(self.chat, "last_user_ts", 0)
            idle_for = time.time() - last
            if idle_for < idle_gate:
                print(f"[curiosity] skip: only idle {idle_for:.0f}s "
                      f"(need {idle_gate}s)", flush=True)
                continue  # someone is talking to her — don't hog the engine
            try:
                self._cycle()
            except Exception as e:
                import traceback
                print(f"[curiosity] cycle failed (next interval retries): {e}",
                      flush=True)
                traceback.print_exc()
