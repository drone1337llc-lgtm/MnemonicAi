"""Zero-dependency .env loader.

Looks for a `.env` file (in the current directory or the project root) and loads
simple KEY=VALUE lines into os.environ without overwriting anything already set.
No third-party packages required.
"""
from __future__ import annotations

import os
from typing import Optional


def load_dotenv(path: Optional[str] = None) -> Optional[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path] if path else [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(here, ".env"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            return p
    return None
