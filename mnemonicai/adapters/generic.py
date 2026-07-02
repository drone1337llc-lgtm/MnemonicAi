"""A concrete, working adapter for a plain perceive/act agent loop.

This is the reference example the platform stubs are modeled on.
"""
from __future__ import annotations

from typing import Callable, List

from ..memory_item import MemoryItem
from .base import MemoryAdapter


class GenericLoopAdapter(MemoryAdapter):
    """Wraps any callable `act(prompt) -> str` with brain-like memory."""

    def run_turn(self, observation: str, goal: str,
                 act: Callable[[str], str], k: int = 6) -> str:
        # 1. perceive the world
        self.on_observation(observation, source="env")
        # 2. recall what's relevant
        recalled: List[MemoryItem] = self.on_before_action(goal, k=k)
        # 3. build the prompt and act
        prompt = self._build_prompt(goal, observation, recalled)
        result = act(prompt)
        # 4. remember the outcome + advance time
        self.on_after_action(result, source="self")
        return result

    @staticmethod
    def _build_prompt(goal: str, observation: str, recalled: List[MemoryItem]) -> str:
        parts = [MemoryAdapter.format_context(recalled),
                 f"\nCurrent observation: {observation}",
                 f"Goal: {goal}\n\nRespond:"]
        return "\n".join(p for p in parts if p)
