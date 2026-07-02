"""Pluggable embedding providers.

`HashingEmbedder` is the default: deterministic, dependency-free, offline. It
uses the hashing trick over unigrams + bigrams to project text into a fixed
L2-normalized vector. It is not semantically deep, but it is more than enough
to drive novelty detection and similarity ranking in demos and tests.

`OpenAICompatibleEmbedder` calls any OpenAI-style `/embeddings` endpoint --
LM Studio, OpenAI, Ollama, etc. -- via stdlib urllib (no `requests` needed).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class HashingEmbedder:
    """Offline, deterministic embedder. No network, no dependencies."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model = "hashing-embedder"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = _tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            h = _hash(tok)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        # bigrams add a little word-order signal
        for a, b in zip(toks, toks[1:]):
            h = _hash(a + "_" + b)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += 0.5 * sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class OpenAICompatibleEmbedder:
    """Embedder backed by an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = None,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/embeddings", data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [row["embedding"] for row in data["data"]]
