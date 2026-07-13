# MnemonicAI / Aerith — Changes

## 2026-07-13 — v1.3.0 "Curiosity"
- **Curiosity engine** (`mnemonicai/curiosity.py`): idle-time autonomy —
  novelty seeking over distant memory pairs, hypothesis forming/testing,
  associative-graph gap hunting, introspection over her own stats. Insights
  scored by embedding information gain; high-gain thoughts stored as
  memories with gain-scaled salience (intrinsic reward) and consolidated by
  sleep-training. Yields to real traffic; emits monitor events.
- **Public endpoint moved to the pod:** my.mnemonicai.org → cloudflared →
  new `omni-proxy.service` (Omni Scale key validation) → `aerith-tunnel`
  → pod MnemonicAI. `llama-omni` retired; GPU0 freed (~0.5GB residual).
- `SYSTEM-CHEATSHEET.md` added: full system reference.

## 2026-07-13 — v1.2.0 "Cloud Brain"
- **Aerith now hosted on RunPod** (A40 48GB): full MnemonicAI stack — memory,
  sleep-training bakes, embedding sidecar, brain monitor — runs pod-side next
  to the raw **f16** weights. Access via SSH tunnel: `localhost:8400` is
  unchanged for the GUI and all agents (`runpod-training/serve/start_tunnel.sh`).
  Local GPUs freed. Local install kept as fallback (`systemctl enable mnemonicai`).
- **Semantic memory retrieval**: Qwen3-Embedding-0.6B sidecar (:8404) replaces
  hashing vectors; existing memories re-embed automatically; hashing fallback
  if the sidecar dies. Recall now matches meaning, not keywords.
- **Test-time self-correction** (`reflect_mode: auto`): substantive non-streamed
  replies get a silent draft → critique → revise pass; agent/tool traffic and
  chit-chat skip it.
- **Speculative decoding investigated**: blocked for hybrid linear-attention
  targets in llama.cpp; config-gated plumbing kept (`draft_gguf`).
- **ARCHITECTURE-ROADMAP.md** added: phased plan to jack-of-all-trades
  (speech, vision, computer use, search, generation) as sidecar organs.

## 2026-07-12 — Aerith v4 "Identity Repair"
- Root-caused "she repeats herself": base model shipped a poisoned chat
  template injecting a foreign "Qwythos/Empero AI" identity into every request,
  plus a trained think-scaffold that restated the question and doubled every
  answer (all 2,160 RunPod rows).
- Fixed: sanitized template (Aerith, created by Sergio Williams — Surge),
  `--reasoning-format deepseek` (think split out of content), cleaned dataset
  (scaffold stripped, 79 varied identity rows), QLoRA repair retrain on A40,
  deployed as v4 under strict fixed naming with sanity gate.
- VRAM-aware stop-start engine swap with rollback (bake crash fix); SIGTERM
  memory persistence + 180s autosave (restart memory-loss fix); train-error
  events surfaced to the monitor.
- Monitor: per-dot organic motion, individual dots in tight groups, music
  shuffle with persistent no-repeat bags, position→frequency (20Hz–20kHz)
  music-reactive dots, ambient synapse pulses.

## In progress — Aerith v5 "Deep Mind"
- Qwen3.6-27B teacher distillation (~3.7k rows), depth upscale 32→48 layers
  (12.4B, direct safetensors — mergekit drops linear_attn tensors), heal
  training, layer-redundancy sparsification, benchmark-gated deploy, fresh
  Q8_0/Q4_K_M quants for local use.
