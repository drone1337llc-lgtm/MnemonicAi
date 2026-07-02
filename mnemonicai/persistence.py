"""SQLite persistence for long-term memory.

Working memory is intentionally *not* persisted -- like the brain, the
short-term workspace is lost between sessions; only consolidated long-term
memory survives a save/load cycle.
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from .memory_item import MemoryItem, MemoryKind

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .memory_system import BrainMemory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT, content TEXT, embedding TEXT,
    created_at REAL, last_access_at REAL, access_count INTEGER,
    salience REAL, strength REAL, stability REAL, activation REAL,
    summary_of TEXT, source TEXT, metadata TEXT
);
CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, weight REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def save(brain: "BrainMemory", path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM links")
        conn.execute("DELETE FROM meta")

        rows = []
        link_rows = []
        for store in (brain.episodic, brain.semantic, brain.procedural):
            for it in store.all():
                rows.append((
                    it.id, it.kind.value, it.content, json.dumps(it.embedding),
                    it.created_at, it.last_access_at, it.access_count,
                    it.salience, it.strength, it.stability, it.activation,
                    json.dumps(it.summary_of), it.source, json.dumps(it.metadata),
                ))
                for dst, w in it.links.items():
                    link_rows.append((it.id, dst, w))

        conn.executemany(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO links VALUES (?,?,?)", link_rows)
        conn.execute("INSERT INTO meta VALUES ('config', ?)",
                     (json.dumps(brain.cfg.to_dict()),))
        conn.execute("INSERT INTO meta VALUES ('clock', ?)", (str(brain.now()),))
        conn.commit()
    finally:
        conn.close()


def load(brain: "BrainMemory", path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT * FROM memories")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(cols, row))
            item = MemoryItem(
                id=r["id"], kind=MemoryKind(r["kind"]), content=r["content"],
                embedding=json.loads(r["embedding"]),
                created_at=r["created_at"], last_access_at=r["last_access_at"],
                access_count=r["access_count"], salience=r["salience"],
                strength=r["strength"], stability=r["stability"],
                activation=r["activation"], summary_of=json.loads(r["summary_of"]),
                source=r["source"], metadata=json.loads(r["metadata"]),
            )
            store = {
                MemoryKind.EPISODIC: brain.episodic,
                MemoryKind.SEMANTIC: brain.semantic,
                MemoryKind.PROCEDURAL: brain.procedural,
            }.get(item.kind)
            if store is not None:
                store.add(item)

        for src, dst, w in conn.execute("SELECT src, dst, weight FROM links"):
            it = (brain.episodic.get(src) or brain.semantic.get(src)
                  or brain.procedural.get(src))
            if it is not None:
                it.links[dst] = w
    finally:
        conn.close()
