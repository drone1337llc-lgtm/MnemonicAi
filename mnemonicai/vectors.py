"""Tiny pure-Python vector math so the package has zero hard dependencies.

If numpy is installed it is used transparently for a speed-up, but nothing
here requires it.
"""
from __future__ import annotations

import math
from typing import List, Sequence

try:  # optional acceleration
    import numpy as _np  # type: ignore
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0 for empty or mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    if _HAVE_NUMPY:
        va = _np.asarray(a, dtype=float)
        vb = _np.asarray(b, dtype=float)
        na = float(_np.linalg.norm(va))
        nb = float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def max_similarity(vec: Sequence[float], others: List[Sequence[float]]) -> float:
    """Highest cosine of `vec` against a list of vectors (0 if list empty)."""
    best = 0.0
    for o in others:
        s = cosine(vec, o)
        if s > best:
            best = s
    return best


def blend(a: Sequence[float], b: Sequence[float], wa: float = 0.5) -> List[float]:
    """Weighted average of two equal-length vectors (used for reconsolidation)."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    wb = 1.0 - wa
    return [wa * x + wb * y for x, y in zip(a, b)]
