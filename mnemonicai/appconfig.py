"""Runtime configuration for the MnemonicAi server (model, ports, training).

Separate from mnemonicai.config.Config (which tunes the *memory dynamics*). This
one governs how the product runs: which model, where its weights are, the LoRA
sleep-training schedule, and the HTTP endpoint.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List


def _default_targets() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]


@dataclass
class AppConfig:
    # ---- server ----
    host: str = "127.0.0.1"
    port: int = 8400

    # ---- model ----
    model_name: str = "ornith-1.0-9b"
    model_path: str = "./models/ornith-1.0-9b"   # HF safetensors dir (trainable)
    gguf_path: str = ""                            # optional GGUF (inference fallback)
    backend: str = "auto"          # auto | transformers | llamacpp | remote | mock
    inference_url: str = ""        # remote OpenAI-compatible engine, e.g.
                                   # "http://192.168.68.36:8401/v1" (llama-server)
    inference_model: str = ""      # model id on the remote engine (default: model_name)
    inference_api_key: str = ""    # bearer token if the remote engine needs one
    # ---- hybrid backend (blue/green local engines + train/convert/swap) ----
    llama_server_exe: str = ""     # path to llama-server (.exe on Windows)
    lora_convert_script: str = ""  # path to llama.cpp's convert_lora_to_gguf.py
    engine_ctx: int = 32768        # llama-server context size; keep modest so
                                   # two engines + QLoRA training share the GPU
    engine_parallel: int = 2       # llama-server slots (each gets ctx/parallel)
    engine_kv_type: str = "q8_0"   # KV cache quantization ("" = f16)
    load_in_4bit: bool = True
    max_new_tokens: int = 384
    max_new_tokens_cap: int = 8192  # hard ceiling on client-requested output
                                    # (prompt + output must fit engine_ctx)
    agent_temperature: float = 0.2  # sampling temp for tool-calling requests
                                    # that don't set their own; low = reliable
                                    # tool formatting (agents send temp=None,
                                    # and the engine default ~0.8 is erratic)
    agent_top_p: float = 0.9
    temperature: float = 0.7
    top_p: float = 0.9

    # ---- persistence ----
    data_dir: str = "./mnemonicai_data"
    memory_db: str = "./mnemonicai_data/memory.db"
    adapter_dir: str = "./mnemonicai_data/adapter"  # LoRA adapter (the "baked" memory)

    # ---- memory / recall ----
    recall_k: int = 6
    perceive_importance: float = 0.6

    # ---- sleep-consolidation training (neocortical consolidation) ----
    train_on_sleep: bool = True
    sleep_every_n_turns: int = 6        # consolidate + train every N chat turns
    train_min_examples: int = 6         # don't train on fewer than this
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=_default_targets)
    train_steps: int = 8
    train_lr: float = 2e-4
    train_batch: int = 1

    # ---- catastrophic-forgetting guards ----
    replay_ratio: float = 0.5             # share of each train batch that is base-capability replay
    eval_holdout: float = 0.2             # fraction of memory examples held out to measure drift
    max_eval_loss_increase: float = 0.15  # roll back if held-out loss rises > this (relative)
    keep_adapter_versions: int = 5        # adapter snapshots retained for rollback

    def ensure_dirs(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.adapter_dir, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str = "config.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        cfg = cls()
        if os.path.isfile(path):
            # utf-8-sig: tolerate the BOM that Windows editors/PowerShell add
            with open(path, "r", encoding="utf-8-sig") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    msg = (f"\n[config] {path} is not valid JSON "
                           f"(line {e.lineno}, column {e.colno}): {e.msg}\n")
                    if "escape" in e.msg.lower():
                        msg += ("[config] Tip: Windows paths in JSON must use "
                                "forward slashes —\n"
                                '         "C:/Users/you/models/…"  not  '
                                '"C:\\Users\\you\\models\\…"\n')
                    msg += ("[config] Fix the file (or delete it to regenerate "
                            "defaults) and re-run.")
                    raise SystemExit(msg) from None
            for k, v in data.items():
                if k in cls.__dataclass_fields__:
                    setattr(cfg, k, v)
        # environment overrides (handy for one-liners and Docker)
        cfg.model_path = os.environ.get("MNEMONICAI_MODEL", cfg.model_path)
        cfg.backend = os.environ.get("MNEMONICAI_BACKEND", cfg.backend)
        cfg.host = os.environ.get("MNEMONICAI_HOST", cfg.host)
        if os.environ.get("MNEMONICAI_PORT"):
            cfg.port = int(os.environ["MNEMONICAI_PORT"])
        if os.environ.get("MNEMONICAI_DATA"):
            cfg.data_dir = os.environ["MNEMONICAI_DATA"]
            cfg.memory_db = os.path.join(cfg.data_dir, "memory.db")
            cfg.adapter_dir = os.path.join(cfg.data_dir, "adapter")
        return cfg
