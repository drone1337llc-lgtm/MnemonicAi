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
        from peft import (LoraConfig, PeftModel, get_peft_model,
                          prepare_model_for_kbit_training)

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
            torch_dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model)

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
    # auto
    try:
        import torch
        if torch.cuda.is_available() and resolve_model_dir(cfg.model_path):
            log(f"[backend] CUDA + weights found → TransformersPeftBackend "
                f"({torch.cuda.get_device_name(0)}).")
            return TransformersPeftBackend(cfg)
        log("[backend] no CUDA or model weights found → MockBackend "
            "(install GPU deps and set model_path for the real model).")
    except Exception as e:
        log(f"[backend] torch unavailable ({e}) → MockBackend.")
    return MockBackend(cfg)
