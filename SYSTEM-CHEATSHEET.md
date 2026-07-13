# Aerith / MnemonicAI — Complete System Cheat Sheet
*Everything the system is, where it lives, and how it got here. (2026-07-13)*

## 1. What Aerith is
- **Model:** Aerith — 9B hybrid linear-attention LLM (Qwen3.5-9B descendant),
  ~120 tok/s decode at Q4, 131,072-token context. Identity: *created by
  Sergio Williams (Surge)* — trained in, template-enforced.
- **Version lineage** (`AERITH_VERSION.json` is the only place versions live;
  paths/names never change so agents never break):
  - v1-v3: SLERP fusion → RunPod QLoRA → local overnight continued-pretraining
  - **v4 (live):** identity/format repair — removed the inherited "Qwythos/
    Empero AI" identity and the think-scaffold that duplicated every answer
    (poisoned chat template + 2,160 scaffolded training rows). Cleaned
    dataset, 79 varied identity rows, A40 QLoRA retrain, sanity-gated deploy.
  - **v5 (in flight):** Qwen3.6-27B teacher distillation (~3.7k rows) →
    depth-upscale 32→48 layers (12.4B, direct safetensors; mergekit silently
    drops this arch's linear_attn tensors — never use it for this) →
    heal-train → layer-redundancy sparsify → benchmark gate vs v4 → deploy.
- **Fixed paths:** HF weights `~/Documents/mergekit/models/Aerith/` (backup
  `Aerith-previous/`); quants `~/Documents/MnemonicAi/models/gguf/
  Aerith-Q8_0.gguf`, `Aerith-Q4_K_M.gguf`, `Aerith-f16.gguf` (backups
  `*.gguf.prev`). Model id on every endpoint: **`Aerith`**.

## 2. Where Aerith runs (cloud, since 2026-07-13)
RunPod A40 48GB pod `mj830tdtoaxntg` (~$0.44/hr) serves **raw f16** with the
full MnemonicAI stack pod-side. Local GPUs are free.

```
you / agents ──► localhost:8400 ──ssh tunnel──► pod MnemonicAI :8400
                                                 ├─ memory + bakes (A40)
my.mnemonicai.org ─► cloudflared ─► omni-proxy   ├─ llama-server engines
                     (:8401, key check)──┘       └─ embed sidecar :8404
```
- **Local access:** `http://localhost:8400` (GUI + OpenAI API, unchanged URLs)
  via systemd `aerith-tunnel.service`.
- **Public access:** `https://my.mnemonicai.org/v1` — `omni-proxy.service`
  validates Omni Scale client keys (`~/.hermes/omni_scale/client_keys.txt`),
  then relays through the tunnel. No key → 401.
- **Pod scripts** (`~/Documents/runpod-training/serve/`): `launch_serve.sh`,
  `deploy_mnemonicai_pod.sh`, `start_tunnel.sh`, `omni_auth_proxy.py`;
  raw-endpoint key in `aerith_api_key.txt` (never committed).
- **Local fallback:** `sudo systemctl enable --now mnemonicai` + hybrid
  backend paths in `config.json` (copy `config.example.json`).

## 3. The MnemonicAI brain (what makes her different)
- **Persistent memory:** every exchange is perceived through a salience gate
  into episodic/semantic/procedural stores with decay, recency, spreading
  activation, and associative links. Autosave every 180s + SIGTERM save.
- **Semantic retrieval:** Qwen3-Embedding-0.6B sidecar — recall matches
  meaning, not keywords ("pet feline" finds the cat memory). Auto re-embeds
  old vectors; hashing fallback if the sidecar dies.
- **Sleep-training (self memory training):** every N turns she consolidates
  and **bakes memories into her own LoRA weights** — replay mix against
  forgetting, eval-loss rollback guard, adapter versioning (`versions/vN`),
  VRAM-aware blue/green or stop-start engine swap, SVG consolidation cards.
- **Curiosity engine (autonomy, new):** when idle ≥3 min she explores on her
  own — novelty seeking (distant memory pairs), hypothesis forming/testing,
  knowledge-graph link walking, introspection over her own stats. Each
  thought is scored by **information gain** (embedding distance from all
  existing memories); high-gain insights become memories with gain-scaled
  salience (intrinsic reward) and flow into sleep-training. Config:
  `curiosity_enabled/every_s/idle_s/gain_floor`.
- **Reflection (test-time compute):** substantive non-streamed replies get a
  silent draft → self-critique → revise pass (`reflect_mode: auto`); skips
  chit-chat and agent/tool traffic.
- **Brain monitor GUI:** live memory dots (per-dot organic motion, individual
  dots in tight clusters), region breathing, synapse pulses, music-reactive
  mode with position→frequency mapping (left 20Hz → right 20kHz), shuffled
  playlists with no-repeat persistence, bake button, curiosity events.

## 4. Endpoints cheat table
| Use | URL | Auth |
|---|---|---|
| GUI + memory-augmented API | `http://localhost:8400` (`/v1`) | none (tunnel-only) |
| Public API (Omni Scale) | `https://my.mnemonicai.org/v1` | Omni client key |
| Embeddings sidecar (pod-internal) | `:8404/v1/embeddings` | none |
| Model id everywhere | `Aerith` | — |
OpenClaw / Hermes / OpenHands: OpenAI-compatible provider, base URL above,
model `Aerith`. Note: scripts hitting the RunPod proxy must send a custom
User-Agent (Python-urllib default gets 403 from RunPod's edge).

## 5. Training / build pipelines (`~/Documents/runpod-training/`)
- `run_runpod_train.sh` + `train_aerith_lora.py`: QLoRA on A40, launch-and-
  detach, `check_training.sh` / `finish_training.sh`, `deploy_v4.sh` pattern
  (merge → version-stamp → quantize → CPU sanity gate → zero-downtime swap).
- `distill/`: Qwen3.6-27B teacher generation (resumable, identity-leak
  filters — the exact contamination path Qwythos used is filtered).
- `build_clean_dataset.py` / `build_train_v5.py`: scaffold stripping,
  identity rows, dedupe.
- Upscale: `mergekit/upscale_direct.py` (direct safetensors self-stack);
  heal: `launch_heal.sh` + `train_heal_12b.py` (includes layer-redundancy
  report for pruning).
- Benchmarks: `mergekit/run_benchmarks.sh` (lm-eval suite, 4-bit, GPU0).

## 6. Repo / release flow
`~/Documents/MnemonicAi` = source of truth, branch `runpod` →
github.com/drone1337llc-lgtm/MnemonicAi → CI builds + publishes Docker to
ghcr.io and Docker Hub on tag push. `config.json` is gitignored (holds the
live key); `config.example.json` is the template. Pre-cloud local snapshot:
`~/Documents/MnemonicAi-local-version/`.

## 7. Roadmap
See `ARCHITECTURE-ROADMAP.md`: sidecar-organ plan for speech-to-speech,
vision, computer use, high-accuracy search, image/video generation — the 12B
core owns language/reasoning/identity/memory; organs are swappable.
