<div align="center">

# 🧠 MnemonicAi

**An LLM that actually remembers — and forgets.**

Self-hosted `ornith-1.0-9b` wrapped in a brain-inspired memory system: a fast hippocampus
that recalls and forgets, and a slow neocortex that **bakes memories into the model's
weights** while it sleeps — with a live cosmic brain monitor so you can watch it think.

![python](https://img.shields.io/badge/python-3.8%2B-4584b6) ![license](https://img.shields.io/badge/license-MIT-57e39a) ![gpu](https://img.shields.io/badge/GPU-NVIDIA%20QLoRA-8b7bff) ![status](https://img.shields.io/badge/core%20deps-zero-ffcb52)

</div>

---

## Why

Today's LLMs either cram everything into the context window or dump it all into a vector
store that never forgets. Neither behaves like a mind. MnemonicAi gives a local model
**two-speed memory, exactly like a brain**:

- **Hippocampus (fast, while attached)** — every message passes a salience gate, lives
  briefly in a decaying working memory, gets recalled and re-injected when relevant, and
  consolidates into **episodic / semantic / procedural** long-term stores. Unreinforced
  memories **fade and are forgotten** (Ebbinghaus decay + spacing effect).
- **Neocortex (slow, in the weights)** — during "sleep," reinforced memories are
  fine-tuned into a **QLoRA adapter**. The model then remembers them **even when
  MnemonicAi isn't running**. No new memories form while detached.

Every request flows through an **OpenAI-compatible API**, so plugging it into OpenClaw,
Hermes, LM Studio, or any OpenAI client is a one-line base-URL change.

## Quick start

```bash
git clone https://github.com/drone1337llc-lgtm/MnemonicAi.git
cd MnemonicAi

# one-file install (installs QLoRA deps, checks CUDA, finds your model)
python3 install.py --model "path-to-cloned-repo/mnemonicai_project/models/ornith-1.0-9b"     # flat safetensors OR HF cache layout

# one-file run (model + memory engine + live brain monitor)
python3 start.py
```

```
══════════════════════════════════════════════════════════════
  MNEMONICAI is live
══════════════════════════════════════════════════════════════
  Brain monitor : http://127.0.0.1:8400/
  OpenAI API    : http://127.0.0.1:8400/v1
  Model         : ornith-1.0-9b   (backend: transformers)
══════════════════════════════════════════════════════════════
```

No GPU or weights yet? `python3 start.py --backend mock` runs everything (API, monitor,
memory engine) with a stubbed model so you can explore.

**Or:** `pip install ".[gpu]"` → `mnemonicai serve` · **Docker:** `docker compose up --build`
· **One-click:** `./run.sh` (macOS/Linux) or `run.bat` (Windows).

## Point any app at it

| Client | Setting |
|---|---|
| OpenClaw / Hermes / any OpenAI SDK | `base_url = http://127.0.0.1:8400/v1`, model `ornith-1.0-9b` |
| `openai` Python | `OpenAI(base_url="http://127.0.0.1:8400/v1", api_key="x")` |
| curl | `POST http://127.0.0.1:8400/v1/chat/completions` |

## The live brain monitor

A real anatomical brain drawn in light over a Milky-Way sky — every dot is a real memory.

- **Three views** — 🧠 **Brain** (memories live on their lobes), ⌬ **Graph**
  (Obsidian-style force graph of memories + Hebbian links; hover, drag, click), ⧖
  **Timeline** (scrubbable session history: births, reinforcements, forgettings, bakes).
  Transitions are physical: memories **explode** out of the brain into the graph and
  **magnetically regroup** onto their lobes on the way back.
- **Click any memory** — an overlay shows its text, strength, recalls, and connections,
  with 📌 pin (never forgotten), ✕ delete, **💾 PNG graph card** and **{} JSON** export.
- **Sound** — subtle synthesized effects for gate/recall/sleep/bake events plus a licensed
  ambient soundtrack; 🔊 mutes everything, 🎵 flips on **music-reactive memories**
  (each memory becomes an equalizer band and pulses to the music with an energetic
  playlist).
- **⛶ Zen mode** — hides the entire UI for a clean view of the brain (press `h`).
- **Zoom/pan everywhere**, a searchable Memories drawer, and a bottom **bake bar** that
  only glows while a LoRA bake is running.

## Consolidation cards

Every successful bake auto-writes a **memory card** — a cosmic SVG snapshot of what was
baked into that adapter version — to `mnemonicai_data/cards/adapter_vN.svg`, alongside
the versioned adapters in `mnemonicai_data/adapter/versions/` (rollback-able).

## Guardrails (catastrophic forgetting)

Sleep-training mixes a **base-capability replay buffer** into every batch, scores a
held-out slice before/after training, **rolls back** any harmful update, and keeps the
last N adapter versions on disk.

## Configuration

Runtime (`config.json`): `port`, `model_path`, `backend` (`auto|transformers|mock`),
`recall_k`, `sleep_every_n_turns`, `train_on_sleep`, LoRA size/steps, replay ratio,
eval-rollback threshold. Memory dynamics (decay τ, salience weights, retrieval blend,
pruning floor) live in `mnemonicai/config.py`. Env overrides: `MNEMONICAI_MODEL`,
`MNEMONICAI_PORT`, `MNEMONICAI_HOST`, `MNEMONICAI_DATA`, `MNEMONICAI_BACKEND`.

## Project layout

```
install.py / start.py     # one-file install · one-file run
monitor.html              # live brain monitor (served at /)
assets/music/             # licensed soundtrack (see MUSIC-LICENSES.md)
mnemonicai/
  server.py               # OpenAI-compatible API + SSE + assets (pure stdlib)
  bridge.py               # per-request memory lifecycle + admin API
  backend.py              # self-hosted QLoRA backend + mock fallback
  trainer.py              # sleep-training + guardrails + consolidation cards
  memory_system.py …      # the memory engine (zero dependencies)
tests/                    # 15 unit tests
```

## Honest notes

- Real weight-baking needs the **HF safetensors** + an **NVIDIA GPU** (a 4080-class card
  is plenty for 9B in 4-bit). GGUF/LM Studio are inference-only — MnemonicAi self-hosts
  inference so freshly baked memories apply instantly.
- Continual LoRA learning is tuned conservatively (small rank, few steps, replay +
  rollback) — it adds memories without cooking the base model, but it is not magic.
- The core engine, server, and monitor have **zero required dependencies**; only the real
  GPU backend needs `requirements-gpu.txt`.

## Music

Licensed tracks via **Artlist** (see [`assets/music/MUSIC-LICENSES.md`](assets/music/MUSIC-LICENSES.md)).
Ambient default: *That's What It Was* (Just for Kicks), *Desire* (Borrtex), *Silent
Transmission* (Tamuz Dekel). Reactive playlist: Giorgio Vitté, Captain Joz, kawauso,
Out of Flux, Yarin Primak, Ziv Moran, Zooki. These tracks are licensed **for this
project only** — do not reuse them outside MnemonicAi.

> Downloaded a code-only bundle? Grab the music packs and unzip them into
> `assets/music/` — the app works fine without them (sound effects are synthesized;
> the footer will read "music files not found").

## Contributing

Issues and PRs welcome. Run `python3 -m unittest discover -s tests` before submitting;
keep the core dependency-free. See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

https://creativecommons.org/licenses/by-nc-sa/4.0/ for the code. Music is separately licensed via Artlist and
is **not** covered by the MIT license.
