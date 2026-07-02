"""Time-dependent memory dynamics: decay and reinforcement.

Key idea: we never mutate `strength` on a timer. Instead `strength` records the
value at `last_access_at`, and the *current* strength is computed on demand by
applying exponential decay from that moment. Reinforcement resets the baseline.
This makes decay exact regardless of how often (or seldom) it is evaluated.
"""
from __future__ import annotations

import math

from .config import Config
from .memory_item import MemoryItem


def tau_of(item: MemoryItem, cfg: Config) -> float:
    """Decay time constant: larger for important and well-stabilized memories."""
    return cfg.tau_base * max(0.1, item.stability) * (0.5 + item.salience)


def current_strength(item: MemoryItem, now: float, cfg: Config) -> float:
    """Ebbinghaus exponential decay from the last access."""
    dt = now - item.last_access_at
    if dt <= 0:
        return item.strength
    return item.strength * math.exp(-dt / tau_of(item, cfg))


def reinforce(item: MemoryItem, now: float, cfg: Config) -> None:
    """Spacing effect / LTP: recall raises strength and grows stability."""
    cur = current_strength(item, now, cfg)
    item.strength = min(1.0, cur + cfg.alpha_reinforce)
    item.stability = item.stability * (1.0 + cfg.beta_stability)
    item.access_count += 1
    item.last_access_at = now


def recency(item: MemoryItem, now: float, cfg: Config) -> float:
    """Recency weight in [0, 1] for retrieval ranking."""
    if item.last_access_at <= 0:
        return 0.0
    dt = max(0.0, now - item.last_access_at)
    return math.exp(-dt / cfg.tau_recency)
