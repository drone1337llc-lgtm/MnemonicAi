#!/usr/bin/env python3
"""QLoRA fine-tuning of a causal LM (Transformers + PEFT + bitsandbytes).

Loads a base model in 4-bit NF4, attaches (or resumes) a LoRA adapter, trains on
a JSONL dataset with a dependency-light manual loop (no accelerate/datasets
required), and saves the adapter with save_pretrained().

Dataset format (JSONL, one object per line) — either:
  {"text": "raw training text"}
  {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
Chat rows are rendered with tokenizer.apply_chat_template().

Usage:
  python qlora_finetune.py --model /path/to/base --data train.jsonl \
      --adapter-out ./adapter [--resume-adapter ./adapter] \
      [--r 16 --alpha 32 --dropout 0.05] [--target-modules q_proj,k_proj,v_proj,o_proj]
      [--all-linear] [--steps 100 | --epochs 1] [--lr 2e-4] [--batch 1]
      [--grad-accum 8] [--max-len 1024] [--eval-holdout 0.1] [--seed 42]

Auth: gated Hub models use the HF_TOKEN environment variable automatically.
Requires: torch (CUDA), transformers>=4.40, peft>=0.10, bitsandbytes>=0.43.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def _require_gpu_stack():
    """Import the heavy stack lazily with a clear message if missing."""
    try:
        import torch  # noqa
        import transformers  # noqa
        import peft  # noqa
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}.\n"
                 "Install: pip install torch transformers peft bitsandbytes accelerate\n"
                 "(CUDA torch wheels: https://pytorch.org — e.g. "
                 "pip install torch --index-url https://download.pytorch.org/whl/cu124)")
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — QLoRA 4-bit training requires an NVIDIA GPU.",
              file=sys.stderr)
    return torch


def load_rows(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        sys.exit(f"No rows found in {path}")
    return rows


def render_text(row: dict, tokenizer) -> str:
    if "messages" in row:
        return tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)
    if "text" in row:
        return row["text"]
    raise ValueError(f"Row needs 'text' or 'messages': keys={list(row)}")


def load_4bit_model(model_path: str, compute_dtype=None):
    """Base model in 4-bit NF4 with double quantization (the QLoRA recipe)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    compute_dtype = compute_dtype or (
        torch.bfloat16 if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported() else torch.float16)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # QLoRA's NormalFloat4
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,      # quantize the quantization constants
    )
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto",
        torch_dtype=compute_dtype)
    return model, tok


def attach_lora(model, resume_adapter=None, r=16, alpha=32, dropout=0.05,
                target_modules=None, use_gradient_checkpointing=True):
    """prepare_model_for_kbit_training + new LoraConfig or resume an adapter."""
    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing)
    if use_gradient_checkpointing:
        # required so gradients flow into the (frozen) quantized base inputs
        model.enable_input_require_grads()
    if resume_adapter and os.path.isfile(os.path.join(resume_adapter, "adapter_config.json")):
        peft_model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        print(f"[resume] continued from adapter: {resume_adapter}")
    else:
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"],
        )
        peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def encode_batch(texts, tokenizer, max_len, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100    # don't learn padding
    return enc["input_ids"], enc["attention_mask"], labels


def make_optimizer(params, lr):
    """Prefer bitsandbytes' paged 8-bit AdamW (QLoRA paper), fall back to AdamW."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr)
    except Exception:
        import torch
        return torch.optim.AdamW(params, lr=lr)


def evaluate(model, tokenizer, rows, max_len, batch) -> float:
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            texts = [render_text(r, tokenizer) for r in rows[i:i + batch]]
            ids, attn, labels = encode_batch(texts, tokenizer, max_len, model.device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            total += float(out.loss); n += 1
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model path or Hub id")
    ap.add_argument("--data", required=True, help="JSONL training file")
    ap.add_argument("--adapter-out", required=True, help="where to save the adapter")
    ap.add_argument("--resume-adapter", default=None, help="existing adapter to continue")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=None,
                    help="comma list, e.g. q_proj,v_proj (default: all attn+MLP proj)")
    ap.add_argument("--all-linear", action="store_true",
                    help="target every linear layer (PEFT 'all-linear')")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (0 = use --epochs)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-holdout", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch = _require_gpu_stack()
    random.seed(args.seed); torch.manual_seed(args.seed)

    rows = load_rows(args.data)
    random.shuffle(rows)
    n_eval = int(len(rows) * args.eval_holdout)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"[data] train={len(train_rows)} eval={len(eval_rows)}")

    model, tok = load_4bit_model(args.model)
    targets = ("all-linear" if args.all_linear else
               ([t.strip() for t in args.target_modules.split(",")]
                if args.target_modules else None))
    model = attach_lora(model, args.resume_adapter, args.r, args.alpha,
                        args.dropout, targets)
    model.config.use_cache = False               # incompatible with checkpointing
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, args.lr)

    steps_per_epoch = max(1, len(train_rows) // (args.batch * args.grad_accum))
    total_steps = args.steps or max(1, int(steps_per_epoch * args.epochs))
    print(f"[train] {total_steps} optimizer steps "
          f"(batch {args.batch} × accum {args.grad_accum})")

    if eval_rows:
        print(f"[eval] pre-train loss: {evaluate(model, tok, eval_rows, args.max_len, args.batch):.4f}")

    step = micro = 0
    while step < total_steps:
        random.shuffle(train_rows)
        for i in range(0, len(train_rows), args.batch):
            texts = [render_text(r, tok) for r in train_rows[i:i + args.batch]]
            ids, attn, labels = encode_batch(texts, tok, args.max_len, model.device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0 or step == total_steps:
                    print(f"  step {step}/{total_steps}  loss {float(out.loss):.4f}")
                if args.save_every and step % args.save_every == 0:
                    ck = os.path.join(args.adapter_out, f"checkpoint-{step}")
                    model.save_pretrained(ck); print(f"  [ckpt] {ck}")
                if step >= total_steps:
                    break

    if eval_rows:
        print(f"[eval] post-train loss: {evaluate(model, tok, eval_rows, args.max_len, args.batch):.4f}")

    model.save_pretrained(args.adapter_out)      # adapter weights + adapter_config.json
    tok.save_pretrained(args.adapter_out)
    print(f"[done] adapter saved → {args.adapter_out}")
    print("Load later with: PeftModel.from_pretrained(base_4bit_model, adapter_dir)")


if __name__ == "__main__":
    main()
