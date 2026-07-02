#!/usr/bin/env python3
"""MnemonicAi — one entry point to run everything locally.

    python3 run.py check              # verify your local LM Studio is reachable
    python3 run.py demo               # offline narrated lifecycle (no LLM, no network)
    python3 run.py story --llm        # same story, but LM Studio does fact-extraction
    python3 run.py chat               # interactive memory REPL (offline)
    python3 run.py chat --llm         # memory-augmented chat via LM Studio (ornith-1.0-9b)

LM Studio settings come from a .env file or environment variables:
    LMSTUDIO_BASE_URL   (default http://192.168.68.36:1010/v1)
    LMSTUDIO_API_KEY
    LMSTUDIO_CHAT_MODEL (default ornith-1.0-9b)
    LMSTUDIO_EMBED_MODEL (optional; else the offline HashingEmbedder is used)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from mnemonicai import BrainMemory, Config, ManualClock
from mnemonicai.adapters import GenericLoopAdapter
from mnemonicai.adapters.lmstudio import build_lmstudio_memory, lmstudio_settings


# --------------------------------------------------------------------------- #
def cmd_check(args) -> int:
    s = lmstudio_settings()
    base = s["base_url"]
    print(f"Checking LM Studio at {base} …")
    req = urllib.request.Request(base + "/models")
    if s["api_key"]:
        req.add_header("Authorization", "Bearer " + s["api_key"])
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        ids = [m.get("id") for m in data.get("data", [])]
        print(f"  ✓ reachable ({(time.time()-t0)*1000:.0f} ms). Models loaded: {ids or '(none)'}")
        if s["chat_model"] not in ids:
            print(f"  ! configured chat model '{s['chat_model']}' is not in the loaded list "
                  f"— load it in LM Studio or set LMSTUDIO_CHAT_MODEL.")
    except Exception as e:
        print(f"  ✗ could not reach LM Studio: {e}")
        print("    Is the LM Studio server started (Developer tab → Start Server) and the")
        print(f"    address correct? Current base_url = {base}")
        return 1

    # tiny generation probe
    try:
        payload = json.dumps({"model": s["chat_model"],
                              "messages": [{"role": "user", "content": "Reply with just: ok"}],
                              "temperature": 0}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if s["api_key"]:
            req.add_header("Authorization", "Bearer " + s["api_key"])
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode())
        txt = out["choices"][0]["message"]["content"].strip()
        print(f"  ✓ chat completion works ({(time.time()-t0)*1000:.0f} ms): “{txt[:60]}”")
    except Exception as e:
        print(f"  ✗ chat completion failed: {e}")
        return 1
    print("All good — you can run:  python3 run.py chat --llm")
    return 0


def cmd_demo(args) -> int:
    import demo
    demo.main()
    return 0


def cmd_story(args) -> int:
    import demo
    if args.llm:
        clock = ManualClock(t0=1_000_000.0)
        mem = build_lmstudio_memory(Config(sleep_every_n_ticks=0), clock=clock)
        print("(running the story with LM Studio doing fact extraction + summaries)\n")
        demo.main(mem, clock)
    else:
        demo.main()
    return 0


def _build_chat_memory(use_llm: bool):
    clock = ManualClock(t0=time.time())
    if use_llm:
        mem = build_lmstudio_memory(Config(sleep_every_n_ticks=12), clock=clock)
    else:
        mem = BrainMemory(Config(sleep_every_n_ticks=12), clock=clock)
    return mem, clock


CHAT_HELP = """\
Type anything to have the agent perceive it. Commands:
  /recall <cue>   retrieve + reinforce memories for a cue
  /sleep          consolidate short-term into long-term (+ prune)
  /stats          show memory counts and sim clock
  /save <path>    save long-term memory to a SQLite file
  /load <path>    load long-term memory from a SQLite file
  /help           show this help
  /quit           exit
"""


def cmd_chat(args) -> int:
    mem, clock = _build_chat_memory(args.llm)
    adapter = GenericLoopAdapter(mem)
    mode = "LM Studio (" + lmstudio_settings()["chat_model"] + ")" if args.llm else "offline"
    print(f"MnemonicAi chat — {mode}. Memory persists across turns and decays over time.")
    print(CHAT_HELP)
    turn = 0
    while True:
        try:
            line = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(" ", 1)
            cmd = parts[0][1:]
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd == "quit":
                break
            if cmd == "help":
                print(CHAT_HELP)
            elif cmd == "recall":
                res = mem.retrieve(arg or "recent", k=5)
                if not res:
                    print("  (nothing relevant)")
                for m in res:
                    print(f"  • [{m.kind.value} s={mem.recall_strength(m):.2f}] {m.content}")
            elif cmd == "sleep":
                print("  consolidation:", mem.sleep())
            elif cmd == "stats":
                print("  ", mem.stats())
            elif cmd == "save":
                mem.save(arg or "agent_memory.db"); print(f"  saved -> {arg or 'agent_memory.db'}")
            elif cmd == "load":
                mem.load(arg or "agent_memory.db"); print(f"  loaded <- {arg or 'agent_memory.db'}")
            else:
                print("  unknown command; /help for options")
            continue

        # a normal user message: perceive -> recall -> (generate) -> remember
        adapter.on_observation(line, source="user", importance=0.6)
        recalled = adapter.on_before_action(line, k=5)
        context = adapter.format_context(recalled)
        if args.llm:
            try:
                sys_msg = ("You are a helpful assistant with long-term memory. "
                           "Use the remembered facts below when relevant; do not invent memories.")
                user_msg = (context + "\n\n" if context else "") + "User: " + line
                reply = mem.llm.chat([{"role": "system", "content": sys_msg},
                                      {"role": "user", "content": user_msg}]).strip()
                print(f"bot › {reply}")
                mem.perceive(reply, source="self", importance=0.4)
            except Exception as e:
                print(f"bot › [LLM error: {e} — run 'python3 run.py check']")
        else:
            if context:
                print("bot › [memory-augmented context injected into the prompt]")
                for m in recalled:
                    print(f"        • [{m.kind.value}] {m.content}")
            else:
                print("bot › [no relevant memories yet — keep talking, then /sleep]")
        mem.tick(dt=1800)  # ~30 min of simulated time per turn
        turn += 1
    print(f"\nSession over. Final memory: {mem.stats()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run the MnemonicAi memory system locally.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("check", help="verify LM Studio connectivity")
    sub.add_parser("demo", help="offline narrated lifecycle")
    ps = sub.add_parser("story", help="narrated lifecycle"); ps.add_argument("--llm", action="store_true")
    pc = sub.add_parser("chat", help="interactive memory REPL"); pc.add_argument("--llm", action="store_true")
    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return 0
    return {"check": cmd_check, "demo": cmd_demo, "story": cmd_story, "chat": cmd_chat}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
