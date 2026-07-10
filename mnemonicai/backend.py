"""Model backends: self-hosted inference + LoRA sleep-training.

Two implementations behind one interface:

* TransformersPeftBackend — the real thing. Loads ornith-1.0-9b (HF safetensors)
  in 4-bit QLoRA on an NVIDIA GPU, generates with a live LoRA adapter, and can
  fine-tune that same adapter on consolidated memories. Because inference and
  training share one in-process adapter, newly baked memories are immediately
  live, and they persist on disk so the model remembers even when MnemonicAi is
  detached. Heavy imports (torch/transformers/peft) happen lazily inside the
  class, so importing this module never requires a GPU.

* MockBackend — dependency-free. Generates simple deterministic replies that
  reflect any injected memory context, and "trains" by bumping an adapter
  version. Lets the whole server + brain monitor run with no GPU and no weights
  (used for local demos and for verifying the pipeline).

Interface both implement:
    generate_stream(messages) -> iterator of text deltas
    generate(messages) -> str
    train(examples) -> dict            # examples: [{"messages":[{role,content}...]}]
    save_adapter(path) / load_adapter(path)
    .name, .adapter_version
"""
from __future__ import annotations

import os
import random
from typing import Dict, Iterator, List


# --------------------------------------------------------------------------- #
def resolve_model_dir(path: str):
    """Find the real loadable model dir inside common layouts.

    Accepts any of:
      * a flat folder containing config.json + *.safetensors
      * a HuggingFace cache folder (blobs/ refs/ snapshots/<hash>/...)
      * a folder containing a nested 'models--org--name' cache dir
    Returns the resolved directory, or None if no loadable model found.
    """
    if not path or not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snaps = os.path.join(path, "snapshots")
    if os.path.isdir(snaps):
        cands = [os.path.join(snaps, d) for d in os.listdir(snaps)]
        cands = [c for c in cands if os.path.isfile(os.path.join(c, "config.json"))]
        if cands:
            return max(cands, key=os.path.getmtime)   # newest snapshot
    try:
        for d in os.listdir(path):
            if d.startswith("models--"):
                r = resolve_model_dir(os.path.join(path, d))
                if r:
                    return r
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- #
class MockBackend:
    name = "mock"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.adapter_version = 0
        self._baked: List[str] = []   # facts "baked into weights" (simulated)

    def _reply(self, messages: List[Dict[str, str]]) -> str:
        sys_ctx = " ".join(m["content"] for m in messages if m.get("role") == "system")
        user = [m["content"] for m in messages if m.get("role") == "user"]
        last = user[-1] if user else ""
        # surface any injected memory so the effect is visible in the demo
        mem_hint = ""
        if "Relevant memories" in sys_ctx or "remember" in sys_ctx.lower():
            snippet = sys_ctx.split("Relevant memories:", 1)[-1].strip().replace("\n", " ")
            if snippet:
                mem_hint = f" (drawing on memory: {snippet[:120]})"
        baked = f" [baked facts: {len(self._baked)}]" if self._baked else ""
        return f"[mock ornith-1.0-9b v{self.adapter_version}] Re: “{last[:80]}”.{mem_hint}{baked}"

    def generate_stream(self, messages, max_new_tokens=None) -> Iterator[str]:
        for w in self._reply(messages).split(" "):
            yield w + " "

    def generate(self, messages, max_new_tokens=None) -> str:
        return "".join(self.generate_stream(messages, max_new_tokens)).strip()

    def train(self, examples: List[dict]) -> dict:
        # "consolidate" the example contents into baked facts; improve a fake loss
        for ex in examples:
            for m in ex.get("messages", []):
                if m.get("role") == "assistant":
                    self._baked.append(m["content"])
        self.adapter_version += 1
        loss = round(1.0 / (1 + 0.15 * self.adapter_version), 4)
        return {"loss": loss, "examples": len(examples), "steps": 0, "baked": len(self._baked)}

    def eval_loss(self, examples: List[dict]) -> float:
        # synthetic: "loss" drifts down as more is consolidated
        return round(1.0 / (1.0 + 0.12 * self.adapter_version), 4)

    def snapshot(self) -> dict:
        return {"version": self.adapter_version, "baked": list(self._baked)}

    def restore(self, snap: dict) -> None:
        self.adapter_version = snap.get("version", self.adapter_version)
        self._baked = list(snap.get("baked", self._baked))

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "mock_adapter.txt"), "w", encoding="utf-8") as f:
            f.write(f"version={self.adapter_version}\nbaked={len(self._baked)}\n")
            for b in self._baked[-500:]:
                f.write(b + "\n")

    def load_adapter(self, path: str) -> None:
        p = os.path.join(path, "mock_adapter.txt")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                lines = f.read().splitlines()
            for ln in lines:
                if ln.startswith("version="):
                    self.adapter_version = int(ln.split("=", 1)[1] or 0)


# --------------------------------------------------------------------------- #
class TransformersPeftBackend:
    """Real self-hosted inference + QLoRA training. Requires torch/transformers/peft."""

    name = "transformers"

    def __init__(self, cfg) -> None:
        import torch  # noqa
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)
        from peft import LoraConfig, PeftModel, get_peft_model

        self.cfg = cfg
        self.torch = torch
        resolved = resolve_model_dir(cfg.model_path)
        if not resolved:
            raise RuntimeError(
                f"Model weights not found at '{cfg.model_path}'. Point model_path at the "
                f"ornith-1.0-9b folder (flat safetensors OR the HF cache with "
                f"blobs/refs/snapshots — both are auto-resolved).")
        if resolved != cfg.model_path:
            print(f"[backend] resolved HF cache layout → {resolved}")
        cfg.model_path = resolved

        quant = None
        if cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, quantization_config=quant, device_map="auto",
            dtype=torch.bfloat16)
        # Targeted k-bit training prep instead of peft's
        # prepare_model_for_kbit_training: that helper upcasts EVERY
        # non-quantized param to fp32, including the ~1B-element embed and
        # lm_head (3.8 GiB each) — which is what OOM'd this box. Only the
        # 1-D norm params actually need fp32 for training stability.
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        adapter_ready = (os.path.isdir(cfg.adapter_dir)
                         and os.path.isfile(os.path.join(cfg.adapter_dir, "adapter_config.json")))
        if adapter_ready:
            self.model = PeftModel.from_pretrained(model, cfg.adapter_dir, is_trainable=True)
        else:
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                              lora_dropout=cfg.lora_dropout, bias="none",
                              task_type="CAUSAL_LM", target_modules=cfg.lora_targets)
            self.model = get_peft_model(model, lcfg)
        self.model.config.use_cache = True
        self.adapter_version = _read_version(cfg.adapter_dir)

    # ---- inference ----
    def generate_stream(self, messages, max_new_tokens=None) -> Iterator[str]:
        import threading
        from transformers import TextIteratorStreamer
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer,
                      max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                      do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p)
        self.model.eval()
        t = threading.Thread(target=self.model.generate, kwargs=kwargs)
        t.start()
        for text in streamer:
            yield text

    def generate(self, messages, max_new_tokens=None) -> str:
        return "".join(self.generate_stream(messages, max_new_tokens)).strip()

    # ---- training (neocortical consolidation) ----
    def train(self, examples: List[dict]) -> dict:
        torch = self.torch
        self.model.train()
        self.model.config.use_cache = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.train_lr)
        losses = []
        for _ in range(self.cfg.train_steps):
            batch = random.sample(examples, min(len(examples), self.cfg.train_batch))
            input_ids, labels, attn = self._encode(batch)
            out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(out.loss.detach()))
        self.model.eval()
        self.model.config.use_cache = True
        self.adapter_version += 1
        return {"loss": round(sum(losses) / max(1, len(losses)), 4),
                "examples": len(examples), "steps": self.cfg.train_steps}

    def _encode(self, batch):
        torch = self.torch
        texts = []
        for ex in batch:
            text = self.tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)
            texts.append(text)
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=1024).to(self.model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"], labels, enc["attention_mask"]

    def eval_loss(self, examples: List[dict]) -> float:
        torch = self.torch
        if not examples:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        bs = max(1, self.cfg.train_batch)
        with torch.no_grad():
            for i in range(0, len(examples), bs):
                input_ids, labels, attn = self._encode(examples[i:i + bs])
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
                total += float(out.loss.detach())
                n += 1
        return round(total / max(1, n), 4)

    def snapshot(self) -> dict:
        from peft import get_peft_model_state_dict
        sd = {k: v.detach().cpu().clone()
              for k, v in get_peft_model_state_dict(self.model).items()}
        return {"version": self.adapter_version, "state": sd}

    def restore(self, snap: dict) -> None:
        from peft import set_peft_model_state_dict
        state = {k: v.to(self.model.device) for k, v in snap["state"].items()}
        set_peft_model_state_dict(self.model, state)
        self.adapter_version = snap["version"]

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        _write_version(path, self.adapter_version)

    def load_adapter(self, path: str) -> None:
        # loading happens in __init__ via PeftModel.from_pretrained
        self.adapter_version = _read_version(path)


# --------------------------------------------------------------------------- #
class RemoteOpenAIBackend:
    """Delegates token generation to a remote OpenAI-compatible server
    (llama-server, LM Studio, vLLM, another MnemonicAi ...) while this
    process keeps running the memory engine. Zero heavy dependencies —
    pure standard library. Training is skipped (do QLoRA sleep-training
    on a box with the transformers backend).
    """

    name = "remote"

    def __init__(self, cfg) -> None:
        import json as _json
        import urllib.request as _rq
        self._json, self._rq = _json, _rq
        self.cfg = cfg
        self.base = (cfg.inference_url or "").rstrip("/")
        if not self.base:
            raise RuntimeError("inference_url is not set in config")
        self.model = cfg.inference_model or cfg.model_name
        self.key = getattr(cfg, "inference_api_key", "") or ""
        self.adapter_version = _read_version(cfg.adapter_dir)
        # probe the remote engine; warn (don't die) if it's not up yet so the
        # memory server survives reboots of the inference box
        try:
            req = self._request(f"{self.base}/models")
            with self._rq.urlopen(req, timeout=8) as r:
                ids = [m.get("id") for m in _json.loads(r.read()).get("data", [])]
            print(f"[backend] remote inference at {self.base} "
                  f"(models: {ids or '(none)'}; using '{self.model}')")
        except Exception as e:
            print(f"[backend] WARNING: remote inference at {self.base} not "
                  f"reachable yet ({e}); requests will fail until it is up.")

    def _request(self, url, payload=None):
        data = self._json.dumps(payload).encode() if payload is not None else None
        req = self._rq.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        return req

    def generate_stream(self, messages, max_new_tokens=None) -> Iterator[str]:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "max_tokens": max_new_tokens or self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
        }
        req = self._request(f"{self.base}/chat/completions", payload)
        with self._rq.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = self._json.loads(data)["choices"][0]["delta"]
                except (KeyError, IndexError, ValueError):
                    continue
                if delta.get("content"):
                    yield delta["content"]

    def generate(self, messages, max_new_tokens=None) -> str:
        return "".join(self.generate_stream(messages, max_new_tokens)).strip()

    def engine_base(self) -> str:
        """Base /v1 URL of the live inference engine. Overridden by the
        hybrid backend to follow blue/green swaps."""
        return self.base

    def proxy_chat(self, payload: dict):
        """Forward a raw OpenAI chat-completions payload straight to the
        engine and return the open urllib response (caller relays it).

        Used for tool-calling / agent requests: the memory pipeline models
        replies as plain text and would drop `tools` and the `tool_calls`
        the client needs to act. The engine handles tools natively, so we
        pass those requests through untouched."""
        payload = dict(payload)
        payload["model"] = self.model
        req = self._request(f"{self.engine_base()}/chat/completions", payload)
        return self._rq.urlopen(req, timeout=600)

    def train(self, examples: List[dict]) -> dict:
        print("[backend] remote backend cannot train; skipping sleep-training "
              "step (memories still consolidate to DB).")
        return {"loss": 0.0, "examples": len(examples), "steps": 0,
                "skipped": "remote backend does not train"}

    def eval_loss(self, examples: List[dict]) -> float:
        return 0.0

    def snapshot(self) -> dict:
        return {"version": self.adapter_version}

    def restore(self, snap: dict) -> None:
        self.adapter_version = snap.get("version", self.adapter_version)

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        _write_version(path, self.adapter_version)

    def load_adapter(self, path: str) -> None:
        self.adapter_version = _read_version(path)


# --------------------------------------------------------------------------- #
class HybridBackend(RemoteOpenAIBackend):
    """The full loop on one box: serve fast (blue/green llama-server pair via
    EngineManager), sleep-train QLoRA on the local GPU (transformers), convert
    the adapter to GGUF, and hot-swap it in with zero client-visible downtime.
    """

    name = "hybrid"

    def __init__(self, cfg) -> None:
        from .hotswap import EngineManager
        self.mgr = EngineManager(cfg)
        cfg.inference_url = self.mgr.ensure_running()
        super().__init__(cfg)

    @property
    def adapter_version(self):
        # always live from the engine state file so every process (server,
        # manual training runs, monitors) agrees on the deployed version
        return self.mgr.state().get("adapter_version", 0)

    @adapter_version.setter
    def adapter_version(self, v):
        pass  # version is owned by hotswap's engine state file

    def engine_base(self) -> str:
        # follow blue/green swaps: always proxy to the live engine
        url = self.mgr.active_url()
        if url:
            self.base = url
        return self.base

    def generate_stream(self, messages, max_new_tokens=None) -> Iterator[str]:
        # refresh routing before every request so swaps take effect instantly
        url = self.mgr.active_url()
        if url:
            self.base = url
        return super().generate_stream(messages, max_new_tokens)

    # ---- the train → convert → swap pipeline ----
    def train(self, examples: List[dict]) -> dict:
        version = self.adapter_version + 1
        print(f"[hybrid] sleep-training v{version} on {len(examples)} examples")

        # 1. QLoRA training on the local GPU (heavy imports stay lazy)
        tb = TransformersPeftBackend(self.cfg)
        try:
            result = tb.train(examples)
            adapter_dir = os.path.abspath(
                os.path.join(self.cfg.adapter_dir, f"v{version}"))
            tb.model.save_pretrained(adapter_dir)
            # transformers 5 exposes NemotronH modules as model.layers.*, but
            # llama.cpp's converter maps the checkpoint's original backbone.*
            # names — rename adapter keys accordingly. NemotronH ONLY: for
            # every other arch (Qwen, Llama, ...) the converter expects the
            # model.* names and the rename breaks tensor mapping.
            st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
            arch = ""
            try:
                import json as _json
                cfg_json = os.path.join(self.cfg.model_path, "config.json")
                with open(cfg_json, encoding="utf-8") as f:
                    arch = " ".join(_json.load(f).get("architectures", []))
            except Exception:
                pass
            if "nemotron" in arch.lower() and os.path.isfile(st_path):
                from safetensors.torch import load_file, save_file
                tensors = load_file(st_path)
                renamed = {
                    k.replace("base_model.model.model.",
                              "base_model.model.backbone.", 1): v
                    for k, v in tensors.items()}
                if renamed.keys() != tensors.keys():
                    save_file(renamed, st_path)
                    print(f"[hybrid] adapter keys renamed for GGUF conversion")
        finally:
            # release VRAM before booting the standby engine
            import gc
            import torch
            del tb
            gc.collect()
            torch.cuda.empty_cache()

        # 2. convert the PEFT adapter to a llama.cpp GGUF adapter
        gguf_adapter = os.path.abspath(
            os.path.join(self.cfg.adapter_dir, f"adapter_v{version}.gguf"))
        import subprocess
        import sys
        conv_env = os.environ.copy()
        # the converter lives in the llama.cpp source tree and imports its
        # sibling `conversion` package and in-repo gguf-py
        repo = os.path.dirname(os.path.abspath(self.cfg.lora_convert_script))
        conv_env["PYTHONPATH"] = os.pathsep.join(
            [repo, os.path.join(repo, "gguf-py"),
             conv_env.get("PYTHONPATH", "")])
        conv = subprocess.run(
            [sys.executable, self.cfg.lora_convert_script,
             "--base", self.cfg.model_path,
             "--outfile", gguf_adapter, adapter_dir],
            capture_output=True, text=True, timeout=600,
            env=conv_env, cwd=repo)
        if conv.returncode != 0 or not os.path.isfile(gguf_adapter):
            raise RuntimeError(
                f"adapter->GGUF conversion failed:\n{conv.stdout[-800:]}"
                f"\n{conv.stderr[-800:]}")

        # 3. blue/green swap (raises and keeps old engine if unhealthy)
        self.mgr.swap_in(gguf_adapter, version)
        _write_version(self.cfg.adapter_dir, version)
        result["adapter_version"] = version
        result["swapped"] = True
        return result


# --------------------------------------------------------------------------- #
class LlamaCppBackend:
    """Fast local inference from a GGUF via llama.cpp (CUDA). No training —
    sleep-consolidation training requires the transformers backend; this
    backend keeps the memory pipeline live while generating 25-100x faster
    than the naive transformers path on Windows (no mamba_ssm kernels).
    """

    name = "llamacpp"

    def __init__(self, cfg) -> None:
        import threading
        # llama.cpp CUDA wheel needs cublas/cudart on the legacy DLL search
        # path; torch ships them, so borrow torch's lib dir.
        try:
            import torch  # noqa: F401
            lib = os.path.join(os.path.dirname(
                __import__("torch").__file__), "lib")
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        from llama_cpp import Llama

        self.cfg = cfg
        self._lock = threading.Lock()   # llama.cpp context is not thread-safe
        print(f"[backend] loading GGUF via llama.cpp: {cfg.gguf_path}")
        self.llm = Llama(model_path=cfg.gguf_path, n_gpu_layers=-1,
                         n_ctx=65536, verbose=False)
        self.adapter_version = _read_version(cfg.adapter_dir)

    # ---- inference ----
    def generate_stream(self, messages, max_new_tokens=None) -> Iterator[str]:
        with self._lock:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens or self.cfg.max_new_tokens,
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                stream=True)
            for part in stream:
                delta = part["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]

    def generate(self, messages, max_new_tokens=None) -> str:
        return "".join(self.generate_stream(messages, max_new_tokens)).strip()

    # ---- training: not supported on GGUF weights ----
    def train(self, examples: List[dict]) -> dict:
        print("[backend] llamacpp backend cannot train LoRA adapters; "
              "skipping sleep-training step (memories still consolidate to DB).")
        return {"loss": 0.0, "examples": len(examples), "steps": 0,
                "skipped": "gguf backend does not train"}

    def eval_loss(self, examples: List[dict]) -> float:
        return 0.0

    def snapshot(self) -> dict:
        return {"version": self.adapter_version}

    def restore(self, snap: dict) -> None:
        self.adapter_version = snap.get("version", self.adapter_version)

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        _write_version(path, self.adapter_version)

    def load_adapter(self, path: str) -> None:
        self.adapter_version = _read_version(path)


# --------------------------------------------------------------------------- #
def _read_version(adapter_dir: str) -> int:
    p = os.path.join(adapter_dir, "mnemonicai_version.txt")
    try:
        with open(p, encoding="utf-8") as f:
            return int(f.read().strip() or 0)
    except Exception:
        return 0


def _write_version(adapter_dir: str, v: int) -> None:
    try:
        with open(os.path.join(adapter_dir, "mnemonicai_version.txt"), "w", encoding="utf-8") as f:
            f.write(str(v))
    except Exception:
        pass


def build_backend(cfg, log=print):
    """Pick a backend: explicit config, else auto-detect CUDA + weights, else Mock."""
    choice = (cfg.backend or "auto").lower()
    if choice == "mock":
        log("[backend] using MockBackend (no GPU needed).")
        return MockBackend(cfg)
    if choice == "transformers":
        return TransformersPeftBackend(cfg)
    if choice == "llamacpp":
        return LlamaCppBackend(cfg)
    if choice == "remote":
        return RemoteOpenAIBackend(cfg)
    if choice == "hybrid":
        return HybridBackend(cfg)
    # auto — check each prerequisite separately so failures are honest
    # Highest preference: a configured remote inference engine.
    if getattr(cfg, "inference_url", ""):
        try:
            return RemoteOpenAIBackend(cfg)
        except Exception as e:
            log(f"[backend] remote inference at {cfg.inference_url} "
                f"unavailable ({e}); trying local backends.")
    # Prefer llama.cpp when a GGUF is configured: on Windows the transformers
    # Mamba path has no CUDA kernels and runs ~3 s/token; llama.cpp runs 75+.
    if cfg.gguf_path and os.path.isfile(cfg.gguf_path):
        try:
            import llama_cpp  # noqa: F401
            log("[backend] GGUF + llama.cpp found → LlamaCppBackend "
                "(fast inference; sleep-training disabled).")
            return LlamaCppBackend(cfg)
        except ImportError:
            log("[backend] GGUF configured but llama-cpp-python not installed; "
                "trying transformers.")
        except Exception:
            import traceback
            log("[backend] LlamaCppBackend FAILED to load — real error below; "
                "trying transformers:")
            log(traceback.format_exc())
    try:
        import torch
    except Exception as e:
        log(f"[backend] torch not importable ({e}) → MockBackend. "
            "Install GPU deps (requirements-gpu.txt) for the real model.")
        return MockBackend(cfg)
    if not torch.cuda.is_available():
        log("[backend] no CUDA device detected → MockBackend "
            "(install a CUDA-enabled torch build for the real model).")
        return MockBackend(cfg)
    if not resolve_model_dir(cfg.model_path):
        log(f"[backend] model weights not found at '{cfg.model_path}' → "
            "MockBackend (set model_path to your ornith-1.0-9b folder).")
        return MockBackend(cfg)
    log(f"[backend] CUDA + weights found → TransformersPeftBackend "
        f"({torch.cuda.get_device_name(0)}).")
    try:
        return TransformersPeftBackend(cfg)
    except Exception:
        import traceback
        log("[backend] TransformersPeftBackend FAILED to load — real error "
            "below; falling back to MockBackend:")
        log(traceback.format_exc())
        # release whatever portion of the model made it onto the GPU before
        # the failure, so the mock fallback doesn't squat on VRAM
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return MockBackend(cfg)
