"""Ready-to-run preset for a local LM Studio server.

LM Studio exposes an OpenAI-compatible API, so this is just a thin factory that
wires the OpenAI-compatible embedder + LLM to your local endpoint.

Environment variables (never hard-code secrets):
    LMSTUDIO_BASE_URL   default: http://192.168.68.36:1010/v1
    LMSTUDIO_API_KEY    your LM Studio key
    LMSTUDIO_CHAT_MODEL default: ornith-1.0-9b
    LMSTUDIO_EMBED_MODEL default: (unset -> use the offline HashingEmbedder)

Usage:
    from mnemonicai.adapters.lmstudio import build_lmstudio_memory
    mem = build_lmstudio_memory()
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import Config
from ..embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from ..env import load_dotenv
from ..llm import OpenAICompatibleLLM
from ..memory_system import BrainMemory

DEFAULT_BASE_URL = "http://192.168.68.36:1010/v1"
DEFAULT_CHAT_MODEL = "ornith-1.0-9b"


def lmstudio_settings() -> dict:
    """Resolve LM Studio settings from a .env file / environment (with defaults)."""
    load_dotenv()
    return {
        "base_url": os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL),
        "api_key": os.environ.get("LMSTUDIO_API_KEY"),
        "chat_model": os.environ.get("LMSTUDIO_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        "embed_model": os.environ.get("LMSTUDIO_EMBED_MODEL"),
    }


def build_lmstudio_memory(config: Optional[Config] = None,
                          clock=None) -> BrainMemory:
    s = lmstudio_settings()
    base_url, api_key = s["base_url"], s["api_key"]
    chat_model, embed_model = s["chat_model"], s["embed_model"]

    llm = OpenAICompatibleLLM(base_url=base_url, model=chat_model, api_key=api_key)

    if embed_model:
        embedder = OpenAICompatibleEmbedder(base_url=base_url, model=embed_model,
                                            api_key=api_key)
    else:
        # No embedding model loaded in LM Studio -> stay fully local for vectors.
        embedder = HashingEmbedder((config or Config()).embedding_dim)

    return BrainMemory(config or Config(), embedder=embedder, llm=llm, clock=clock)
