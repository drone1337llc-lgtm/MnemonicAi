"""The framework-agnostic integration contract.

A `MemoryAdapter` maps four lifecycle hooks onto a host agent framework. The
memory core never imports any framework; adapters are the only glue.
"""
from __future__ import annotations

from typing import List

from ..memory_item import MemoryItem
from ..memory_system import BrainMemory


class MemoryAdapter:
    """Base adapter. Subclass and wire these hooks into your agent's loop."""

    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory

    def on_observation(self, content: str, source: str = "env", **kw) -> None:
        """Call whenever the agent perceives something (env, user, tool result)."""
        self.memory.perceive(content, source=source, **kw)

    def on_before_action(self, cue: str, k: int = 6) -> List[MemoryItem]:
        """Call before the agent reasons/acts. Returns memories to inject."""
        return self.memory.retrieve(cue, k=k)

    def on_after_action(self, result: str, source: str = "self", **kw) -> None:
        """Call after the agent acts, to remember the outcome."""
        self.memory.perceive(result, source=source, **kw)
        self.memory.tick()

    def on_session_end(self, save_path: str = None) -> dict:
        """Call at the end of a session to consolidate and (optionally) persist."""
        report = self.memory.sleep()
        if save_path:
            self.memory.save(save_path)
        return report

    # convenience: render retrieved memories as prompt context
    @staticmethod
    def format_context(memories: List[MemoryItem]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for m in memories:
            lines.append(f"- ({m.kind.value}) {m.content}")
        return "\n".join(lines)
