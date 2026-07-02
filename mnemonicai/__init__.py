"""MnemonicAi -- a brain-inspired memory system for LLM agents.

Short-term (working) memory that decays, an attention gate that decides what
is worth encoding, an offline consolidation ('sleep') pass, and long-term
episodic / semantic / procedural stores with Ebbinghaus decay, spacing-effect
reinforcement, and Hebbian associative links.

Runs fully offline by default; plug in any OpenAI-compatible endpoint
(LM Studio, OpenAI, Ollama) for real embeddings and reasoning.
"""
from __future__ import annotations

from .config import Config
from .memory_item import MemoryItem, MemoryKind
from .memory_system import BrainMemory, ManualClock
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .llm import HeuristicLLM, OpenAICompatibleLLM

__version__ = "2.2.0"
__all__ = [
    "Config",
    "MemoryItem",
    "MemoryKind",
    "BrainMemory",
    "ManualClock",
    "HashingEmbedder",
    "OpenAICompatibleEmbedder",
    "HeuristicLLM",
    "OpenAICompatibleLLM",
]
