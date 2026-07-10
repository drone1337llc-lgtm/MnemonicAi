#!/usr/bin/env python3
"""Adapter utilities: inspect, snapshot/restore state dicts, find target modules.

Subcommands
  inspect   ADAPTER_DIR            — show adapter_config.json + tensor stats
  targets   --model PATH           — list Linear module names (target_modules candidates)
  snapshot  --model P --adapter D --out state.pt
                                   — get_peft_model_state_dict → CPU .pt snapshot
  restore   --model P --adapter D --state state.pt --out DIR
                                   — set_peft_model_state_dict from snapshot, save
  vram      --params 9e9 [--ctx 1024]
                                   — rough QLoRA memory estimate

The snapshot/restore pair is the in-process rollback pattern: capture adapter
weights before a risky training pass, restore if evaluation regresses.

Auth: gated Hub models use the HF_TOKEN environment variable automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_inspect(args):
    cfg_path = os.path.join(args.adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"No adapter_config.json in {args.adapter_dir}")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    print("== adapter_config.json ==")
    for k in ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules",
              "bias", "task_type", "base_model_name_or_path", "use_rslora", "use_dora"):
        if k in cfg:
            print(f"  {k}: {cfg[k]}")
    # tensor stats without loading the base model
    try:
        from safetensors import safe_open
        st = os.path.join(args.adapter_dir, "adapter_model.safetensors")
        if os.path.isfile(st):
            total = 0
            with safe_open(st, framework="pt") as f:
                keys = list(f.keys())
                for k in keys:
                    shape = f.get_slice(k).get_shape()
                    n = 1
                    for s in shape:
                        n *= s
                    total += n
            print(f"== adapter_model.safetensors ==\n  tensors: {len(keys)}"
                  f"\n  parameters: {total:,} (~{total*2/1e6:.1f} MB at fp16)")
    except ImportError:
        print("  (pip install safetensors for tensor stats)")


def cmd_targets(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. pip install torch transformers")
    print("[targets] loading model structure (meta device — no weights)…")
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    cfg = AutoConfig.from_pretrained(args.model)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    names = {}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("Linear", "Linear8bitLt", "Linear4bit"):
            names[name.split(".")[-1]] = names.get(name.split(".")[-1], 0) + 1
    print("Linear module leaf-names (use as LoraConfig.target_modules):")
    for leaf, cnt in sorted(names.items(), key=lambda x: -x[1]):
        print(f"  {leaf:<16} ×{cnt}")
    print('Tip: target_modules="all-linear" targets every Linear except the LM head.')


def _load_peft(model_path, adapter_dir, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto", torch_dtype=dtype)
    return PeftModel.from_pretrained(base, adapter_dir, is_trainable=trainable)


def cmd_snapshot(args):
    import torch
    from peft import get_peft_model_state_dict
    model = _load_peft(args.model, args.adapter, trainable=False)
    sd = {k: v.detach().cpu().clone()
          for k, v in get_peft_model_state_dict(model).items()}
    torch.save(sd, args.out)
    n = sum(v.numel() for v in sd.values())
    print(f"[snapshot] {len(sd)} tensors, {n:,} params → {args.out}")


def cmd_restore(args):
    import torch
    from peft import set_peft_model_state_dict
    model = _load_peft(args.model, args.adapter, trainable=True)
    sd = torch.load(args.state, map_location="cpu")
    set_peft_model_state_dict(model, sd)
    model.save_pretrained(args.out)
    print(f"[restore] adapter restored from {args.state} → saved {args.out}")


def cmd_vram(args):
    p = float(args.params)
    weights = p * 0.55 / 1e9          # ≈0.55 bytes/param for NF4 + double-quant
    lora = p * 0.0015 * 2 / 1e9       # adapters (r=16-ish) fp16 + grads, rough
    opt = lora * 2                    # AdamW moments (paged 8-bit is less)
    act = args.ctx * 0.75 / 1024      # very rough activation/KV headroom per 1k ctx
    total = weights + lora + opt + act + 0.8
    print(f"QLoRA rough VRAM for {p/1e9:.1f}B params @ ctx {args.ctx}:")
    print(f"  4-bit weights ≈ {weights:.1f} GB\n  adapters+grads ≈ {lora+opt:.2f} GB")
    print(f"  activations/overhead ≈ {act+0.8:.1f} GB\n  TOTAL ≈ {total:.1f} GB")
    print("(estimate only — batch, ctx and checkpointing dominate the margin)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("adapter_dir"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("targets"); s.add_argument("--model", required=True); s.set_defaults(fn=cmd_targets)
    s = sub.add_parser("snapshot")
    s.add_argument("--model", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--out", default="adapter_state.pt"); s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("restore")
    s.add_argument("--model", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--state", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_restore)
    s = sub.add_parser("vram")
    s.add_argument("--params", required=True, help="e.g. 9e9 for a 9B model")
    s.add_argument("--ctx", type=int, default=1024); s.set_defaults(fn=cmd_vram)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
