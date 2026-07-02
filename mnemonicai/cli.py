"""Console entry point: `mnemonicai serve` (also used by start.py).

Boots the self-hosted model + memory engine + live brain monitor and serves the
OpenAI-compatible API. A background thread broadcasts state periodically so the
monitor animates decay even when idle.
"""
from __future__ import annotations

import argparse
import atexit
import os
import threading
import time
import webbrowser

from .appconfig import AppConfig
from .config import Config
from .events import EventBus
from .backend import build_backend
from .memory_system import BrainMemory
from .trainer import SleepTrainer
from .bridge import MemoryChat
from .server import App, serve


def launch(cfg: AppConfig, open_browser: bool = True) -> int:
    cfg.ensure_dirs()
    bus = EventBus()
    print(f"[mnemonicai] starting backend '{cfg.backend}' …")
    backend = build_backend(cfg)

    mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=time.time)
    if os.path.isfile(cfg.memory_db):
        try:
            mem.load(cfg.memory_db)
            print(f"[mnemonicai] loaded long-term memory from {cfg.memory_db}")
        except Exception as e:
            print(f"[mnemonicai] could not load memory: {e}")
    try:
        backend.load_adapter(cfg.adapter_dir)
    except Exception:
        pass

    trainer = SleepTrainer(mem, backend, bus, cfg)
    chat = MemoryChat(mem, backend, bus, trainer, cfg)
    app = App(cfg, bus, chat, backend)
    httpd = serve(app)

    def _save():
        try:
            mem.save(cfg.memory_db)
            print(f"\n[mnemonicai] saved long-term memory → {cfg.memory_db}")
        except Exception as e:
            print(f"[mnemonicai] save failed: {e}")
    atexit.register(_save)

    # periodic state heartbeat so the monitor shows live decay when idle
    def _heartbeat():
        while True:
            time.sleep(3.0)
            try:
                chat.publish_state()
            except Exception:
                pass
    threading.Thread(target=_heartbeat, daemon=True).start()

    url = f"http://{cfg.host}:{cfg.port}/"
    bar = "═" * 62
    print("\n" + bar)
    print("  MNEMONICAI is live")
    print(bar)
    print(f"  Brain monitor : {url}")
    print(f"  OpenAI API    : {url}v1     ← point OpenClaw / Hermes / LM Studio here")
    print(f"  Model         : {cfg.model_name}   (backend: {backend.name})")
    print(f"  Memory        : {cfg.memory_db}")
    print(f"  Adapter       : {cfg.adapter_dir}  (baked-in memory, v{getattr(backend,'adapter_version',0)})")
    print(bar + "\n")
    if backend.name == "mock":
        print("  NOTE: MOCK backend (no GPU/model). Install GPU deps and set --model")
        print("        to your ornith-1.0-9b weights for the real model.\n")

    if open_browser and cfg.host in ("127.0.0.1", "localhost"):
        threading.Timer(1.0, lambda: _try_open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mnemonicai] shutting down …")
    finally:
        httpd.shutdown()
        _save()
    return 0


def _try_open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _apply_args(cfg: AppConfig, args) -> AppConfig:
    if getattr(args, "port", None):
        cfg.port = args.port
    if getattr(args, "host", None):
        cfg.host = args.host
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    if getattr(args, "model", None):
        cfg.model_path = args.model
    if getattr(args, "sleep_every", None):
        cfg.sleep_every_n_turns = args.sleep_every
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mnemonicai",
                                description="An LLM that remembers (and forgets).")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the server + memory + live brain monitor")
    for parser in (p, s):  # accept flags with or without the 'serve' subcommand
        parser.add_argument("--port", type=int)
        parser.add_argument("--host")
        parser.add_argument("--backend")
        parser.add_argument("--model")
        parser.add_argument("--sleep-every", type=int, dest="sleep_every")
        parser.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    cfg = _apply_args(AppConfig.load("config.json"), args)
    return launch(cfg, open_browser=not getattr(args, "no_browser", False))


if __name__ == "__main__":
    raise SystemExit(main())
