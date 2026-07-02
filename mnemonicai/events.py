"""A tiny thread-safe pub/sub event bus.

The memory engine publishes events (gate, working_add, consolidate, train, …);
the HTTP server fans them out to every connected Server-Sent-Events client so
the live brain monitor can animate what the engine is actually doing.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List


class EventBus:
    def __init__(self, history: int = 300) -> None:
        self._subs = set()
        self._lock = threading.Lock()
        self._history: List[dict] = []
        self._hmax = history

    def publish(self, event: dict) -> None:
        if "t" not in event:
            event["t"] = round(time.time(), 3)
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._hmax:
                self._history = self._history[-self._hmax:]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # a slow client shouldn't stall the engine

    def subscribe(self) -> "queue.Queue[dict]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        with self._lock:
            self._subs.discard(q)

    def recent(self) -> List[dict]:
        with self._lock:
            return list(self._history)
