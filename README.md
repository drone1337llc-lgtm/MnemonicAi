# MnemonicAI — Complete Walkthrough

MnemonicAI is a **memory-native AI system**: a language model that runs entirely on
your own hardware, remembers what you tell it across sessions, consolidates those
memories while it "sleeps," and can **bake** them permanently into its own weights.

The model it serves today is **Aria** — a 9B-parameter fusion model (see
"The model" below). Everything is watchable live in the **brain monitor**, a
visualization of memories as living dots inside a brain.

---


## ☁️ Cloud GPU deployment (current, since 2026-07-13)

Aria now runs on a RunPod A40 (48GB) at **raw f16 precision** — the full
MnemonicAI stack (memory, sleep-training bakes, embedding sidecar, monitor)
lives pod-side next to the weights. This folder is the source of truth for
code; a pre-cloud snapshot lives at `~/Documents/MnemonicAi-local-version`.

- **Access (GUI + API, unchanged URLs):** `runpod-training/serve/start_tunnel.sh`
  forwards `localhost:8400` → pod. Agents keep using `http://<this-pc>:8400/v1`.
- **Pod scripts:** `runpod-training/serve/` — `launch_serve.sh` (raw f16
  llama-server pod), `deploy_mnemonicai_pod.sh` (full stack + memory
  migration), `start_tunnel.sh`. API key: `serve/aerith_api_key.txt` (never
  committed; `config.json` is gitignored — copy `config.example.json`).
- **Local fallback:** `sudo systemctl enable --now mnemonicai` with
  `"backend": "hybrid"` + local paths in `config.json`.
- **Public endpoint:** `https://my.mnemonicai.org/v1` — Cloudflare tunnel →
  local `omni-proxy.service` (validates Omni Scale client keys) → SSH tunnel
  → pod. systemd: `aerith-tunnel.service`, `omni-proxy.service`.
- **Curiosity engine:** when idle, Aria explores autonomously — novelty
  seeking, hypothesis testing, knowledge-graph walks, introspection — scored
  by information gain; novel insights become memories and feed sleep-training.
- See `SYSTEM-CHEATSHEET.md` for the complete system reference, `CHANGES.md`
  for the release log, and `ARCHITECTURE-ROADMAP.md` for the sidecar plan.

## Table of Contents

1. [Quick start](#1-quick-start)
2. [What's running: the architecture](#2-whats-running-the-architecture)
3. [The model (Aria)](#3-the-model-aerith)
4. [The brain monitor](#4-the-brain-monitor)
5. [Talking to it: the API](#5-talking-to-it-the-api)
6. [Memory: how it works](#6-memory-how-it-works)
7. [Baking memories into weights](#7-baking-memories-into-weights)
8. [Swapping models with zero downtime](#8-swapping-models-with-zero-downtime)
9. [Running as a service](#9-running-as-a-service)
10. [Configuration reference](#10-configuration-reference)
11. [Script reference](#11-script-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Quick start

Everything already installed on this box? Then it's just:

```bash
# it's probably already running as a service:
curl http://localhost:8400/health
# → {"status": "ok", "model": "Aria", "backend": "hybrid", ...}

# open the brain monitor:
xdg-open http://localhost:8400/
```

Fresh start (manual run instead of the service):

```bash
cd ~/Documents/MnemonicAi
python3 start.py                    # real model + memory + monitor
python3 start.py --backend mock     # UI only, no GPU needed
```

Point any OpenAI-compatible client (OpenClaw, Hermes, LM Studio, curl…) at
`http://localhost:8400/v1`. No API key is required (any placeholder works).

---

## 2. What's running: the architecture

```
                 you / your agents
                        │
             http://localhost:8400
                        │
        ┌───────────────┴─────────────────┐
        │        MnemonicAI server        │   mnemonicai/server.py
        │  OpenAI API + monitor + memory  │
        └───────┬───────────────┬─────────┘
                │               │
        memory engine     hybrid backend
     (perceive/recall/    (proxies chat to
      sleep/forget)        llama-server)
                │               │
        mnemonicai_data/   blue/green engine pair
          memory.db        ports 8402 / 8403
          adapter/         (llama.cpp, CUDA,
          cards/            one active at a time)
```

- **The server** (`:8400`) is a single Python process (systemd service
  `mnemonicai`). It owns the memory engine and never restarts during model
  or adapter swaps.
- **Inference** runs in a compiled `llama-server` binary (llama.cpp, CUDA)
  on the RTX 3090. Two slots exist (ports 8402/8403); exactly one is live.
- **Training** (sleep-consolidation QLoRA) runs in-process with
  transformers/PEFT on the same GPU, then converts the adapter to GGUF and
  swaps it into the engine.
- **Swaps are VRAM-aware**: if a second engine fits in free VRAM, it's a
  zero-downtime blue/green flip. If not, the old engine is stopped first and
  the new one started (a brief pause — and it rolls back automatically if
  the new engine fails to boot).

---

## 3. The model (Aria)

Aria is a 9B model served as a Q4_K_M GGUF. Canonical locations (fixed
names — version is metadata, not filename, so automation never breaks):

| What | Where |
|---|---|
| Full HF model | `~/Documents/mergekit/models/Aerith/` |
| Version/lineage metadata | `.../Aerith/AERITH_VERSION.json` |
| Serving quant (Q4_K_M) | `~/Documents/MnemonicAi/models/gguf/Aerith-Q4_K_M.gguf` |
| High-quality quant (Q8_0) | `~/Documents/MnemonicAi/models/gguf/Aerith-Q8_0.gguf` |
| Previous version backups | `Aerith-previous/`, `Aerith-Q4_K_M.gguf.prev` |

Current lineage (v3): SLERP fusion of Aerith-9B-Final + MnemonicAI-tuned
ornith → QLoRA fine-tune on RunPod → QLoRA continued-pretraining (Wikipedia +
RedPajama, local overnight run). Check `AERITH_VERSION.json` for the
authoritative record.

---

## 4. The brain monitor

Open `http://localhost:8400/`. What you're looking at:

- **Regions** — sensory cortex, prefrontal, hippocampus, temporal,
  cerebellum. Each memory kind lives in its anatomical home (episodic →
  temporal, semantic → prefrontal, procedural → cerebellum).
- **Dots** — every dot is one memory. Size/brightness track strength; dots
  drift on their own little orbits, twinkle independently, and crowd into
  tight (but individual) swarms as a region fills. Gold ring = pinned.
- **Particles/pulses** — thoughts in flight: perceive events stream from
  sensory → hippocampus; recalls pulse hippocampus → the memory. The brain
  also fires quiet ambient synapse traffic on its own.
- **Gold bar** (bottom) — a LoRA bake in progress.

**Controls** (bottom bar):
- *Teach the brain a memory* — inject text directly (with importance slider)
- *Recall cue* — trigger retrieval and watch what lights up
- *Sleep* — run consolidation now
- *Bake to weights ✦* — run sleep-training (QLoRA) now; see section 7
- *Memories / Graph* — list view, or the Obsidian-style force graph of
  memories and their Hebbian links (`#graph` / `#timeline` in the URL work too)
- 🔊 music, 🎵 music-*reactive* mode (the brain pulses to the audio) — each
  mode shuffles through its own playlist
- Zoom with the scroll wheel, drag to pan, `m` to mute

---

## 5. Talking to it: the API

OpenAI-compatible; the model name is `Aria`:

```bash
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Aria","messages":[{"role":"user","content":"hi"}]}'
```

Every conversation turn feeds the memory engine automatically (perceive →
working memory → consolidation). Other endpoints:

| Endpoint | What it does |
|---|---|
| `GET /health` | status, live model, adapter version |
| `GET /api/state` | current memory snapshot |
| `GET /events` | SSE stream (what the monitor watches) |
| `POST /api/perceive` | `{"text": "...", "importance": 0..1}` inject a memory |
| `POST /api/recall` | `{"cue": "..."}` retrieve |
| `POST /api/sleep` | consolidate now |
| `POST /api/train` | bake to weights now |
| `POST /admin/swap-base` | swap the served base model (see §8) |
| `POST /admin/apply-adapter` | install an externally-trained adapter GGUF |

⚠️ There is **no authentication** and the server listens on all interfaces —
anyone on your LAN/tailnet can use it, including the admin endpoints.

---

## 6. Memory: how it works

- **Perceive** — every user message (or `/api/perceive`) passes a salience
  gate; admitted items enter working memory.
- **Sleep** — every `sleep_every_n_turns` turns (or on demand) working
  memories consolidate into long-term stores (episodic/semantic/procedural),
  form gists and Hebbian links, and weak memories decay. Pinned memories
  never decay.
- **Persistence** — long-term memory is saved to
  `mnemonicai_data/memory.db` on shutdown (including systemd stop/restart)
  **and autosaved every 3 minutes**, so restarts and crashes don't lose it.
- **Recall** — retrieval blends semantic similarity, strength, and recency;
  recalled memories are reinforced (and visible as pulses in the monitor).

---

## 7. Baking memories into weights

"Baking" = sleep-training: a QLoRA pass that teaches the model its own
memories so recall survives **even with the memory system detached**.

Runs automatically during sleep (when there are ≥ `train_min_examples`
memories) or on demand via the monitor's *Bake to weights* button.

Pipeline: build training examples from the strongest memories → mix in
base-capability replay (anti-catastrophic-forgetting) → QLoRA train on the
GPU → convert adapter to GGUF → swap into the live engine (VRAM-aware, §8)
→ snapshot to `mnemonicai_data/adapter/versions/vN`.

Guardrails:
- **Eval rollback** — a held-out slice is scored before/after; a harmful
  update is rolled back automatically.
- **Version history** — the last `keep_adapter_versions` adapters are kept;
  any can be restored.
- **Consolidation cards** — every accepted bake writes an SVG snapshot to
  `mnemonicai_data/cards/` showing exactly what was baked.

If a bake fails, the monitor now shows the error (and the engine keeps
serving the previous adapter — a failed bake never takes the service down).

---

## 8. Swapping models with zero downtime

To serve a new/updated model:

```bash
cd ~/Documents/MnemonicAi
# from a ready GGUF (give the HF dir too so sleep-training tracks the new base):
./mn_swap_model.sh /path/to/model.gguf /path/to/hf-model-dir
# or straight from an HF safetensors dir (converts + quantizes first):
./mn_swap_model.sh /path/to/hf-model-dir
```

What happens: the standby engine slot boots the new model → health check →
traffic flips atomically → old engine retires after 60s. If free VRAM can't
hold two engines at once, it automatically falls back to stop-then-start
(seconds of downtime) and **rolls back to the old engine if the new one
fails**. The memory database is untouched; the memory *adapter* resets
(LoRA is tied to the weights it was trained on) and sleep-training rebuilds
it from the DB.

---

## 9. Running as a service

```bash
sudo systemctl status mnemonicai      # is it up?
sudo systemctl restart mnemonicai     # safe: memory saves on stop
sudo journalctl -u mnemonicai -f      # live logs
```

The unit is installed by `mn_service.sh`. The engine processes (llama-server)
are children managed by the server — don't start/stop them by hand; use the
swap endpoints/scripts.

---

## 10. Configuration reference

`config.json` (working copy in this directory; loaded at startup):

| Key | Meaning |
|---|---|
| `model_name` | display/API name (`Aria`) |
| `model_path` | HF safetensors dir (used by sleep-training) |
| `gguf_path` | GGUF the engine serves |
| `backend` | `hybrid` (serve via llama.cpp + train via transformers) |
| `llama_server_exe` / `lora_convert_script` | llama.cpp binary + converter |
| `engine_ctx` / `engine_parallel` / `engine_kv_type` | engine context size, slots, KV quant |
| `max_new_tokens_cap` | clamp on client `max_tokens` |
| `agent_temperature` / `agent_top_p` | sampling used for agent traffic |
| `sleep_every_n_turns` | auto-consolidation cadence |
| `train_min_examples` / `train_steps` / `train_lr` / `train_batch` | bake hyperparams |
| `replay_ratio` / `eval_holdout` / `max_eval_loss_increase` | anti-forgetting guardrails |
| `keep_adapter_versions` | adapter history depth |
| `lora_r` / `lora_alpha` / `lora_dropout` / `lora_targets` | LoRA shape |

---

## 11. Script reference

| Script | Purpose |
|---|---|
| `start.py` | run everything (used by the service) |
| `mn_swap_model.sh` | zero-downtime model swap (GGUF or HF dir) |
| `mn_run.sh` | start/stop/status helper for manual runs |
| `mn_install.sh` | full from-scratch install (deps, venv, model checks) |
| `mn_service.sh` | install/uninstall the systemd unit |
| `mn_adapter.sh` | list/restore baked adapter versions |
| `mn_backup.sh` | back up memory DB + adapters |
| `mn_diff.sh` | diff two adapter versions |
| `install.py` | write config.json (e.g. `--model /path/to/Aerith`) |
| `demo.py` / `serve_sim.py` / `simulator.html` | no-GPU demos |
| `qlora_finetune.py` / `qlora_generate.py` / `qlora_adapter_tools.py` | standalone QLoRA utilities (outside the sleep loop) |

---

## 12. Troubleshooting

**Bake starts, GPU goes idle, no new adapter** — the old failure mode: the
freshly-baked engine couldn't fit next to the live one in VRAM and died on
boot. Fixed: the swap now detects this and stop-starts instead. If a bake
still fails, the monitor shows the error; check
`mnemonicai_data/engine_8402.log` / `engine_8403.log`.

**Monitor shows no activity** — refresh the page (the event stream doesn't
survive server restarts). Also: no traffic = a calm brain; only real
perceive/recall/sleep events animate strongly.

**`/health` says `"backend": "mock"`** — the real model failed to load;
check the journal. The mock serves canned responses so the UI stays usable.

**Memories vanished after a crash** — the DB autosaves every 3 minutes and
on any clean stop; at most you lose the last ~3 minutes. Baked memories are
in the weights regardless (`mnemonicai_data/cards/` shows what was baked).

**Disk full** — big artifacts (model checkpoints, datasets) belong on the
4TB drive (`/media/surge/4 Tera Storage1/Projects`), not the root disk.
`docker system df` and `du -sh ~/Documents/mergekit/models/*` are the usual
suspects.

**Engine OOM on the 3090** — the engine + KV cache (131k ctx, 2 slots,
q8_0) uses ~9-16GB depending on load; sleep-training temporarily adds a
4-bit copy of the model. If boots fail, lower `engine_ctx` or
`engine_parallel`.

## Licensing & attribution

Copyright © 2026 Sergio Williams ("Surge"). "MnemonicAI" and "Aria" are trademarks of Sergio Williams.

- **Source code** — source-available, **non-commercial** (personal/educational/research). Commercial use needs a separate license. See [`LICENSE`](LICENSE).
- **Aria model weights & original content** (docs, site copy, media) — **CC BY-NC 4.0**. See [`LICENSE-CONTENT`](LICENSE-CONTENT). Aria derives from open Qwen-family weights (Apache-2.0).
- **Third-party components** (every model, library, dataset, font, service — with authors and licenses) — [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).
- **Background music** — licensed via Artlist; per-track credits in [`assets/music/MUSIC-LICENSES.md`](assets/music/MUSIC-LICENSES.md).

The live site's public Licenses, Privacy, Terms, Refunds, Security, and Contact pages mirror these. Commercial licensing / questions: support@mnemonicai.org.
