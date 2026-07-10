#!/usr/bin/env python3
"""Generate with a 4-bit base model + LoRA adapter(s) — and compare or merge.

Covers the PEFT inference lifecycle:
  * load base in 4-bit NF4, attach one or more adapters (PeftModel.from_pretrained
    + load_adapter/set_adapter)
  * --compare: same prompt with the adapter ENABLED vs DISABLED
    (model.disable_adapter()) to see exactly what the adapter learned
  * --merge-out DIR: merge_and_unload() the adapter into full-precision weights
    and save a standalone model (re-loads base unquantized for a clean merge)

Usage:
  python qlora_generate.py --model BASE --adapter DIR "your prompt"
  python qlora_generate.py --model BASE --adapter DIR --compare "prompt"
  python qlora_generate.py --model BASE --adapter A1 --adapter A2 --use A2 "prompt"
  python qlora_generate.py --model BASE --adapter DIR --merge-out ./merged

Auth: gated Hub models use the HF_TOKEN environment variable automatically.
"""
from __future__ import annotations

import argparse
import os
import sys


def _stack():
    try:
        import torch  # noqa
        from transformers import AutoModelForCausalLM  # noqa
        from peft import PeftModel  # noqa
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. "
                 "pip install torch transformers peft bitsandbytes accelerate")
    import torch
    return torch


def load_base(model_path, four_bit=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = dict(device_map="auto", torch_dtype=dtype)
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tok


def attach_adapters(model, adapter_dirs, use=None):
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(
        model, adapter_dirs[0], adapter_name=os.path.basename(adapter_dirs[0]) or "adapter0")
    for d in adapter_dirs[1:]:
        peft_model.load_adapter(d, adapter_name=os.path.basename(d) or d)
    if use:
        peft_model.set_adapter(use)
        print(f"[adapter] active: {use}")
    return peft_model


def generate(model, tok, prompt, max_new=256, temperature=0.7):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:                    # no chat template on base LMs
        text = prompt
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-4), top_p=0.9,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def merge_adapter(base_path, adapter_dir, out_dir):
    """merge_and_unload: bake the adapter into standalone full-precision weights.

    Note: merging into a 4-bit quantized base is lossy/unsupported — reload the
    base UNquantized (fp16/bf16) for the merge. Needs enough RAM/VRAM for that.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype = torch.float16
    print("[merge] loading base un-quantized (this needs full-model memory)…")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype,
                                                device_map="auto")
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    print(f"[merge] standalone merged model → {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", action="append", default=[],
                    help="adapter dir (repeatable for multi-adapter)")
    ap.add_argument("--use", default=None, help="adapter name to activate")
    ap.add_argument("--compare", action="store_true",
                    help="generate with adapter enabled vs disabled")
    ap.add_argument("--merge-out", default=None,
                    help="merge adapter into base and save standalone model here")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("prompt", nargs="?", default=None)
    args = ap.parse_args()

    _stack()

    if args.merge_out:
        if not args.adapter:
            sys.exit("--merge-out requires --adapter")
        merge_adapter(args.model, args.adapter[0], args.merge_out)
        return

    if not args.prompt:
        sys.exit("Provide a prompt (or use --merge-out).")

    model, tok = load_base(args.model, four_bit=True)
    if args.adapter:
        model = attach_adapters(model, args.adapter, args.use)
    model.eval()

    if args.compare and args.adapter:
        print("=== WITH adapter ===")
        print(generate(model, tok, args.prompt, args.max_new, args.temperature))
        print("\n=== WITHOUT adapter (disable_adapter) ===")
        with model.disable_adapter():
            print(generate(model, tok, args.prompt, args.max_new, args.temperature))
    else:
        print(generate(model, tok, args.prompt, args.max_new, args.temperature))


if __name__ == "__main__":
    main()
