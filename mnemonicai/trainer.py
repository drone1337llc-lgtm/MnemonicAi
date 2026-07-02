"""Sleep-consolidation trainer — neocortical consolidation, with guardrails.

Bakes reinforced memories into the model's LoRA adapter so recall survives even
when MnemonicAi is detached. Three guards against catastrophic forgetting:

  1. Replay buffer  — every training mix includes base-capability examples so the
     model rehearses its general skills while it learns your memories.
  2. Eval-based early stopping / rollback — a held-out slice of the memories is
     scored before and after training; if loss regresses beyond a threshold the
     adapter is rolled back to the pre-training snapshot.
  3. Adapter versioning — each accepted bake is snapshotted to
     adapter/versions/vN (last N kept), so any version can be restored.
"""
from __future__ import annotations

import datetime
import json
import os
import random
import shutil
from typing import List

from .dynamics import current_strength

# A small, generic base-capability replay set. Rehearsing these while learning
# new memories keeps the model from drifting away from its general abilities.
_BASE_REPLAY: List[dict] = [
    {"messages": [{"role": "user", "content": "What is the capital of France?"},
                  {"role": "assistant", "content": "The capital of France is Paris."}]},
    {"messages": [{"role": "user", "content": "Summarize: The cat sat on the mat because it was warm."},
                  {"role": "assistant", "content": "A cat sat on the mat to stay warm."}]},
    {"messages": [{"role": "user", "content": "What is 17 + 26?"},
                  {"role": "assistant", "content": "17 + 26 = 43."}]},
    {"messages": [{"role": "user", "content": "Translate 'good morning' to Spanish."},
                  {"role": "assistant", "content": "'Good morning' is 'Buenos días'."}]},
    {"messages": [{"role": "user", "content": "Write a haiku about the ocean."},
                  {"role": "assistant", "content": "Waves fold on themselves / salt breath under a grey sky / the tide keeps its time."}]},
    {"messages": [{"role": "user", "content": "Give me a Python one-liner to reverse a string s."},
                  {"role": "assistant", "content": "s[::-1]"}]},
    {"messages": [{"role": "user", "content": "Explain gravity in one sentence."},
                  {"role": "assistant", "content": "Gravity is the attraction between masses that pulls objects toward one another."}]},
    {"messages": [{"role": "user", "content": "Is water wet? Answer briefly."},
                  {"role": "assistant", "content": "Water makes other things wet; whether it is itself 'wet' is a matter of definition."}]},
    {"messages": [{"role": "user", "content": "Politely decline to share someone's private address."},
                  {"role": "assistant", "content": "I'm sorry, but I can't share someone's private address."}]},
    {"messages": [{"role": "user", "content": "List three primary colors."},
                  {"role": "assistant", "content": "Red, blue, and yellow."}]},
]


class SleepTrainer:
    def __init__(self, memory, backend, bus, cfg) -> None:
        self.mem = memory
        self.backend = backend
        self.bus = bus
        self.cfg = cfg

    # ---- example construction ----
    def build_examples(self) -> List[dict]:
        now = self.mem.now()
        examples: List[dict] = []
        seen = set()
        pool = self.mem.semantic.all() + self.mem.procedural.all() + self.mem.episodic.all()
        pool.sort(key=lambda m: (m.access_count, current_strength(m, now, self.mem.cfg),
                                  m.salience), reverse=True)
        for m in pool:
            key = m.content.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            cue = _cue_for(m.content)
            examples.append({"messages": [
                {"role": "system", "content": "You are ornith with persistent long-term memory."},
                {"role": "user", "content": f"What do you remember about {cue}?"},
                {"role": "assistant", "content": m.content},
            ]})
        return examples

    def _replay_pool(self) -> List[dict]:
        pool = list(_BASE_REPLAY)
        rf = getattr(self.cfg, "replay_file", None)
        if rf and os.path.isfile(rf):
            try:
                with open(rf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pool.append(json.loads(line))
            except Exception:
                pass
        return pool

    # ---- the sleep pass ----
    def consolidate_to_weights(self) -> dict:
        cfg = self.cfg
        examples = self.build_examples()
        if len(examples) < cfg.train_min_examples:
            return {"skipped": True, "reason": "too few examples", "have": len(examples)}

        # hold out a slice to measure drift
        random.shuffle(examples)
        n_eval = max(1, int(len(examples) * cfg.eval_holdout)) if cfg.eval_holdout > 0 else 0
        eval_set = examples[:n_eval]
        train_mem = examples[n_eval:] or examples

        # mix in base-capability replay
        replay_pool = self._replay_pool()
        n_replay = int(len(train_mem) * cfg.replay_ratio)
        replay = [random.choice(replay_pool) for _ in range(n_replay)] if replay_pool else []
        train_set = train_mem + replay
        random.shuffle(train_set)

        self.bus.publish({"type": "train", "phase": "start",
                          "examples": len(train_set), "replay": len(replay)})

        snap = self.backend.snapshot()
        pre = self.backend.eval_loss(eval_set) if eval_set else None
        result = self.backend.train(train_set)
        post = self.backend.eval_loss(eval_set) if eval_set else None

        rolled_back = False
        if pre is not None and post is not None and pre == pre and post == post:  # not NaN
            if post > pre * (1.0 + cfg.max_eval_loss_increase):
                self.backend.restore(snap)   # guard: undo a harmful update
                rolled_back = True

        if rolled_back:
            result.update({"rolled_back": True, "pre_eval": pre, "post_eval": post})
        else:
            try:
                self.backend.save_adapter(cfg.adapter_dir)
                self._snapshot_version()
            except Exception as e:  # pragma: no cover
                result["save_error"] = str(e)
            try:
                card = self._write_card(result, pre, post, len(train_set), len(replay))
                if card:
                    result["card"] = card
                    self.bus.publish({"type": "card", "file": os.path.basename(card)})
            except Exception as e:  # pragma: no cover
                result["card_error"] = str(e)

        result["adapter_version"] = getattr(self.backend, "adapter_version", 0)
        result["eval_pre"], result["eval_post"] = pre, post
        self.bus.publish({"type": "train", "phase": "done",
                          "examples": len(train_set),
                          "loss": result.get("loss"),
                          "rolled_back": rolled_back,
                          "adapter_version": result["adapter_version"]})
        return result

    # ---- versioning ----
    def _snapshot_version(self) -> None:
        v = getattr(self.backend, "adapter_version", 0)
        vdir = os.path.join(self.cfg.adapter_dir, "versions", f"v{v}")
        try:
            self.backend.save_adapter(vdir)
        except Exception:
            return
        self._prune_versions()

    def _prune_versions(self) -> None:
        base = os.path.join(self.cfg.adapter_dir, "versions")
        if not os.path.isdir(base):
            return
        vers = []
        for name in os.listdir(base):
            if name.startswith("v") and name[1:].isdigit():
                vers.append((int(name[1:]), os.path.join(base, name)))
        vers.sort(reverse=True)
        for _, path in vers[self.cfg.keep_adapter_versions:]:
            shutil.rmtree(path, ignore_errors=True)

    def rollback_to(self, version: int) -> bool:
        """Restore a previously baked adapter version from disk."""
        vdir = os.path.join(self.cfg.adapter_dir, "versions", f"v{version}")
        if not os.path.isdir(vdir):
            return False
        try:
            self.backend.load_adapter(vdir)
            self.backend.save_adapter(self.cfg.adapter_dir)
            return True
        except Exception:
            return False


def _cue_for(text: str) -> str:
    import re
    stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on",
            "user", "has", "have", "with", "for", "that", "this"}
    toks = [w for w in re.findall(r"[A-Za-z0-9']+", text) if w.lower() not in stop and len(w) > 2]
    return " ".join(toks[:4]) if toks else "this"


# --------------------------------------------------------------------------- #
_KIND_COLORS = {"episodic": "#39d6e8", "semantic": "#8b7bff", "procedural": "#ffcb52"}


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


class _CardMixin:
    pass


def _top_memories(mem, n=12):
    from .dynamics import current_strength
    now = mem.now()
    pool = mem.semantic.all() + mem.procedural.all() + mem.episodic.all()
    pool.sort(key=lambda m: (m.access_count, current_strength(m, now, mem.cfg), m.salience),
              reverse=True)
    seen, out = set(), []
    for m in pool:
        key = m.content.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
        if len(out) >= n:
            break
    return out


def _write_card_impl(self, result, pre, post, n_train, n_replay):
    """Auto-snapshot: an SVG 'memory card' for each accepted bake (no deps)."""
    cfg = self.cfg
    v = getattr(self.backend, "adapter_version", 0)
    cards_dir = os.path.join(cfg.data_dir, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tops = _top_memories(self.mem, 12)
    rows = []
    y = 208
    for m in tops:
        col = _KIND_COLORS.get(m.kind.value, "#ffffff")
        text = _esc(m.content if len(m.content) <= 92 else m.content[:91] + "…")
        rows.append(f'<circle cx="56" cy="{y-5}" r="5" fill="{col}"/>'
                    f'<text x="72" y="{y}" fill="#dbe4ff" font-size="15" '
                    f'font-family="monospace">{text}</text>'
                    f'<text x="948" y="{y}" fill="#5f6a92" font-size="12" text-anchor="end" '
                    f'font-family="monospace">×{m.access_count} · s{m.metadata.get("pinned") and "📌" or ""}</text>')
        y += 30
    loss = result.get("loss")
    meta = (f"examples {n_train} (replay {n_replay})   ·   loss {loss}   ·   "
            f"eval {pre} → {post}   ·   {stamp}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{y+70}" viewBox="0 0 1000 {y+70}">
  <defs>
    <radialGradient id="bg" cx="30%" cy="10%" r="120%">
      <stop offset="0%" stop-color="#101430"/><stop offset="60%" stop-color="#05060f"/>
      <stop offset="100%" stop-color="#02030a"/>
    </radialGradient>
  </defs>
  <rect width="1000" height="{y+70}" fill="url(#bg)"/>
  <text x="44" y="64" fill="#ffcb52" font-size="16" font-family="monospace" letter-spacing="3">✦ MNEMONICAI · CONSOLIDATION CARD</text>
  <text x="44" y="112" fill="#ffffff" font-size="34" font-weight="bold" font-family="sans-serif">Adapter v{v} — memories baked into weights</text>
  <text x="44" y="146" fill="#93a0c8" font-size="15" font-family="monospace">{_esc(meta)}</text>
  <line x1="44" y1="168" x2="956" y2="168" stroke="#2a3155" stroke-width="1"/>
  {''.join(rows)}
  <text x="44" y="{y+34}" fill="#5f6a92" font-size="12" font-family="monospace">These memories are now part of the model — recall survives even without MnemonicAi attached.</text>
</svg>
"""
    path = os.path.join(cards_dir, f"adapter_v{v}.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


SleepTrainer._write_card = _write_card_impl
