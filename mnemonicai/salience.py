"""The attention / salience gate.

Scores each incoming item 0..1. Only items above `salience_threshold` are
admitted into working memory; everything else is forgotten immediately, just
like the overwhelming majority of sensory input.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .config import Config
from .vectors import cosine, max_similarity

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
              "or", "in", "on", "at", "for", "with", "as", "by", "it", "this",
              "that", "has", "have", "had", "will", "can", "be", "um", "ok",
              "yes", "no", "so", "well", "just", "about"}


def _informative_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def content_richness(text: str) -> float:
    """0..1 information content. Near-empty utterances ('ok.') score ~0, so they
    can't ride high novelty into memory on an empty store."""
    distinct = set(_informative_tokens(text))
    return min(1.0, len(distinct) / 3.0)


# A tiny affective lexicon. In production, let the LLM tag emotional charge.
_EMOTION_WORDS = {
    "love", "hate", "fear", "afraid", "angry", "furious", "joy", "happy",
    "sad", "grief", "excited", "terrified", "shocked", "amazing", "awful",
    "danger", "dangerous", "urgent", "critical", "emergency", "died", "death",
    "win", "lost", "failure", "success", "pain", "hurt", "allergic", "warning",
}


def emotion_score(text: str) -> float:
    toks = text.lower().split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t.strip(".,!?;:") in _EMOTION_WORDS)
    # saturating: a couple of emotional words already means "charged"
    return min(1.0, hits / 2.0)


class SalienceScorer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def score(
        self,
        embedding: Sequence[float],
        text: str,
        existing_embeddings: List[Sequence[float]],
        importance: float = 0.0,
        goal_embedding: Optional[Sequence[float]] = None,
        emotion: Optional[float] = None,
        explicit: float = 0.0,
    ) -> float:
        s, _ = self.score_detailed(embedding, text, existing_embeddings,
                                   importance, goal_embedding, emotion, explicit)
        return s

    def score_detailed(
        self,
        embedding: Sequence[float],
        text: str,
        existing_embeddings: List[Sequence[float]],
        importance: float = 0.0,
        goal_embedding: Optional[Sequence[float]] = None,
        emotion: Optional[float] = None,
        explicit: float = 0.0,
    ):
        """Return (salience, breakdown) so callers can report the components."""
        cfg = self.cfg
        # Novelty only counts to the extent the input actually carries content:
        # this stops empty/filler utterances from passing the gate on an empty
        # store (where everything looks maximally novel).
        richness = content_richness(text)
        nov_raw = 1.0 - max_similarity(embedding, existing_embeddings)
        novelty = nov_raw * richness
        emo = emotion_score(text) if emotion is None else emotion
        goal = cosine(embedding, goal_embedding) if goal_embedding else 0.0
        goal = max(0.0, goal)  # negative similarity shouldn't reward
        s = (cfg.w_nov * novelty
             + cfg.w_imp * max(0.0, min(1.0, importance))
             + cfg.w_emo * emo
             + cfg.w_goal * goal
             + explicit)
        s = max(0.0, min(1.0, s))
        breakdown = {"novelty": round(nov_raw, 3), "emotion": round(emo, 3),
                     "goal": round(goal, 3), "richness": round(richness, 3)}
        return s, breakdown
