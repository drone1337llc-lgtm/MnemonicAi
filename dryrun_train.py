"""Dry-run of the full sleep-training pipeline.

Runs the identical code path the server uses (HybridBackend.train), with
synthetic memories, then reports.
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mnemonicai.appconfig import AppConfig
from mnemonicai.backend import HybridBackend

cfg = AppConfig.load(os.path.join(BASE_DIR, "config.json"))
cfg.train_steps = 4  # keep the dry-run quick

be = HybridBackend(cfg)
print(f"[dryrun] engine active: {be.mgr.active_url()} "
      f"(v{be.adapter_version})", flush=True)

examples = [
    {"messages": [
        {"role": "user", "content": "What GPU serves Aria's inference?"},
        {"role": "assistant", "content": "Aria's inference runs on the RTX 3090 in the AIassistant PC."}]},
    {"messages": [
        {"role": "user", "content": "Who is Surge?"},
        {"role": "assistant", "content": "Surge is my human. He runs a three-PC home AI cluster."}]},
    {"messages": [
        {"role": "user", "content": "What is MnemonicAi?"},
        {"role": "assistant", "content": "MnemonicAi is the brain-inspired memory system that wraps my model."}]},
]

t0 = time.time()
result = be.train(examples)
print(f"[dryrun] train+convert+swap finished in {time.time()-t0:.0f}s", flush=True)
print("[dryrun] result:", __import__('json').dumps(result), flush=True)
print("[dryrun] engine now:", be.mgr.active_url(),
      __import__('json').dumps(be.mgr.state()), flush=True)
print("DRYRUN_OK" if result.get("swapped") else "DRYRUN_NO_SWAP", flush=True)
