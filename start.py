#!/usr/bin/env python3
"""MnemonicAi — the one file to run.

    python3 start.py                      # self-hosted model + memory + live monitor
    python3 start.py --backend mock       # try the UI with no GPU/model
    python3 start.py --model /path/to/ornith-1.0-9b --port 8400

Starts an OpenAI-compatible API and the live brain monitor. Point OpenClaw /
Hermes / LM Studio / any client at http://127.0.0.1:8400/v1 and open
http://127.0.0.1:8400/ to watch the brain.
"""
from mnemonicai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
