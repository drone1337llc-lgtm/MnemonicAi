"""Optional host-platform adapter templates.

There is no public API for OpenClaw or Hermes, and none is required: the
recommended way to integrate MnemonicAi into ANY agent is `GenericLoopAdapter`
(see generic.py) — call its four hooks from your agent's loop and you are done.

These two classes are thin, non-raising templates you can copy if you later
want a named adapter for a specific framework. They simply reuse the base
hooks; `register(host)` prints guidance instead of failing, so importing this
module never breaks a local run.
"""
from __future__ import annotations

from .base import MemoryAdapter

_MAPPING = (
    "Attach the four hooks to your framework's event loop:\n"
    "  observation/message  -> self.on_observation(text, source=...)\n"
    "  before prompt build  -> self.format_context(self.on_before_action(goal))\n"
    "  after action/result  -> self.on_after_action(result)\n"
    "  session close        -> self.on_session_end(save_path=...)"
)


class OpenClawAdapter(MemoryAdapter):
    """Template for an OpenClaw-style agent. Reuses the standard four hooks."""

    def register(self, host=None) -> None:
        print("[OpenClawAdapter] No API needed. " + _MAPPING)


class HermesAdapter(MemoryAdapter):
    """Template for a Hermes-style agent. Reuses the standard four hooks."""

    def register(self, host=None) -> None:
        print("[HermesAdapter] No API needed. " + _MAPPING)
