"""Central configuration for the MnemonicAi memory system.

Every tunable knob in the system lives here so behavior is reproducible and
easy to sweep. Defaults are chosen so that a demo shows *visible* encoding,
decay, and consolidation on a human-readable timescale.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- Working (short-term) memory ----
    working_capacity: int = 7          # Miller's 7 +/- 2
    activation_decay: float = 0.35     # fraction of activation lost per tick
    rehearse_boost: float = 0.40       # activation added when an item is rehearsed
    wm_stage_threshold: float = 0.05   # below this activation, item leaves WM for staging

    # ---- Salience (encoding gate) ----
    salience_threshold: float = 0.35   # minimum salience to enter working memory
    w_nov: float = 0.40                # weight: novelty
    w_imp: float = 0.30                # weight: explicit importance
    w_emo: float = 0.20                # weight: emotional charge
    w_goal: float = 0.10               # weight: relevance to current goal

    # ---- Decay (Ebbinghaus forgetting curve) ----
    tau_base: float = 3 * 24 * 3600.0  # base decay time constant, seconds (3 days)

    # ---- Reinforcement (spacing effect / LTP) ----
    alpha_reinforce: float = 0.25      # strength added per successful recall
    beta_stability: float = 0.60       # stability growth factor per recall

    # ---- Consolidation (sleep) ----
    theta_consolidate: float = 0.25    # strength*salience bar to reach long-term
    k_rehearsals: int = 3              # rehearsals/recalls that force consolidation
    dup_threshold: float = 0.92        # cosine above which items merge into a gist
    link_increment: float = 0.20       # Hebbian link weight added on co-activation
    link_cap: float = 1.0              # maximum link weight

    # ---- Retrieval ranking ----
    tau_recency: float = 24 * 3600.0   # recency time constant, seconds (24h)
    w_sim: float = 0.50                # weight: cue-memory cosine similarity
    w_rec: float = 0.20                # weight: recency
    w_str: float = 0.20                # weight: current strength
    w_assoc: float = 0.10              # weight: spreading activation
    retrieval_floor: float = 0.05      # minimum score to be returned

    # ---- Pruning / eviction ----
    strength_floor: float = 0.05       # prune long-term items below this strength
    keep_access_count: int = 2         # never prune items recalled at least this often
    archive_on_evict: bool = False     # if True, pruned items go to a cold archive list

    # ---- Autonomic sleep cadence ----
    sleep_every_n_ticks: int = 20      # auto-consolidate every N ticks (0 = manual only)

    # ---- Embeddings ----
    embedding_dim: int = 256           # dimensionality for the offline HashingEmbedder

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
