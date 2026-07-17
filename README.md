# MnemonicAI

**A private, memory-native AI assistant.** Aria remembers what you tell her — your
projects, your preferences, the way you think — and gets more useful every time you
talk. Every conversation builds on the last instead of starting from zero, and your
memories live in an instance that is yours alone.

MnemonicAI is the whole system: the model, the brain-modeled memory engine, the
tenant gateway, the live brain monitor, and the voice. This repository is the
source of truth for all of it.

---

## Aria and Aria2

| | **Aria** | **Aria2** |
|---|---|---|
| Tiers | Trial & Starter | Pro |
| Model | Aria-14B (Qwen3-14B + SFT) | The same Aria-14B, full precision |
| Precision | Quantized (GGUF) | bf16 |
| Compute | Shared serving | Dedicated GPU pod, just for you |
| Adaptation | Per-user memory + sleep-training bakes | Everything Aria has **plus T² per-user adaptation** |

**What T² actually is (the honest paragraph):** Aria2 uses T²
(Transformer² / SVF — *singular-value fine-tuning*, from Sakana AI's
Transformer² work) to make one shared set of weights behave like *your* model.
Instead of retraining or storing a full copy of the network per user, T² learns
tiny per-tenant "expert vectors" that scale the singular values of the weight
matrices — kilobytes per user, not gigabytes — and composes them into the model
at serve time. The result: your Aria2 instance genuinely adapts to you, cheaply
enough to give every Pro user their own. It is an efficiency technique layered
on an open base model, not magic, and we say so.

Aria descends from **Qwen3-14B** (Apache-2.0, by Alibaba's Qwen team), merged
and fine-tuned by us. The project openly stands on excellent open work — see
[`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) for every model, library, dataset, font,
and service we build on.

---

## Architecture

```
            users / agents
                  │
        ┌─────────┴──────────┐
        │   Tenant gateway   │  cloud/gateway.py  (:8700)
        │  webapp · auth ·   │  signup / auth / chat / monitor /
        │  billing · uploads │  upload / Stripe webhooks
        └───┬────────────┬───┘
            │            │
   per-tenant brains   serving split
   (isolated memory,  ┌──────────────────────────────┐
    one DB per user)  │ shared: llama.cpp GGUF engine │  Trial / Starter
            │         │ dedicated: bf16 pod + T² vecs │  Pro (Aria2)
            │         └──────────────────────────────┘
     memory pipeline: perceive → salience gate → working
     memory → consolidate (sleep) → bake into weights
```

- **Memory pipeline** — every message passes a hippocampus-style salience gate;
  what matters enters working memory, consolidates into episodic/semantic/
  procedural long-term stores (with Hebbian links and Ebbinghaus-style decay),
  and is retrieved by meaning via a Qwen3-Embedding sidecar. Forgetting is a
  feature: unreinforced noise fades.
- **Sleep-training / weight-baking** — reinforced memories are periodically
  trained into a versioned LoRA adapter ("baking"), so recall survives even
  with the memory system detached. Guardrails: base-capability replay against
  catastrophic forgetting, held-out eval with automatic rollback, and full
  adapter version history.
- **Blue/green hybrid llama.cpp backend** — inference runs in a compiled
  `llama-server` (CUDA); two engine slots exist and swaps are VRAM-aware: if a
  second engine fits, the flip is zero-downtime; if not, stop-then-start with
  automatic rollback if the new engine fails to boot. The memory engine lives
  in the long-running server process and never restarts during swaps.
- **Tenant gateway + tiers** — `cloud/gateway.py` fronts the SaaS: signup
  (instant no-card trial, or Stripe checkout for paid), access-key auth
  (keys stored only as hashes), per-tenant isolated memory DBs, quota-enforced
  and ClamAV-scanned codebase uploads (scanner-unavailable fails **closed**),
  and Stripe webhooks for tier lifecycle. Public tiers: **Trial** $0 (14 days,
  no card, 1 GB) · **Starter** $19/mo (5 GB, shared) · **Pro** $79/mo (20 GB,
  dedicated Aria2 pod). Paid tiers are free for 15 days; first charge on day 15.
- **Isolation & privacy, honestly stated** — every tenant has their own memory
  database and upload area; a request authenticated as tenant A has no code path
  to tenant B's data, and the brain monitor stream is tenant-scoped. Storage is
  per-tenant isolated and access-key-authenticated; **at-rest encryption is on
  the near-term roadmap** (we describe what's built, not what sounds good).
- **Brain monitor** — a live visualization of the memory system: each dot is a
  memory, color-coded by kind; perceive/recall events and bakes animate in real
  time. In the cloud webapp it is scoped strictly to your own instance.
- **Voice** — Aria speaks with a Qwen3-TTS "Jessica" voice (Apache-2.0 base),
  served as a sidecar. Speech-to-text and speech-to-speech bridging are on the
  roadmap (`ARCHITECTURE-ROADMAP.md`).
- **Serverless / dedicated split** — shared tiers ride a pooled quantized
  engine; Pro spins up a dedicated GPU pod running full-precision Aria-14B with
  the tenant's T² vectors composed in. Pod lifecycle is managed by
  `cloud/podlife.py`.
- **Curiosity engine & self-reflection** — when idle, Aria explores her own
  knowledge (novelty seeking, hypothesis testing, knowledge-graph walks) scored
  by information gain; insights become memories and feed sleep-training.
  Self-reflection passes check her own work on hard problems.

---

## Quickstart (docker compose)

```bash
git clone https://github.com/drone1337llc-lgtm/MnemonicAi
cd MnemonicAi
cp config.example.json config.json     # point model paths at your weights
docker compose up --build
```

- The stack serves an OpenAI-compatible API and the brain monitor on
  `http://localhost:8400` (`GET /health` to check; `xdg-open http://localhost:8400/`
  for the monitor). The adapter UI runs on `:8401`.
- GPU required for real inference (the compose file uses the NVIDIA runtime);
  `python3 start.py --backend mock` runs the UI with canned responses, no GPU.
- Prebuilt images are published by CI to
  `ghcr.io/drone1337llc-lgtm/mnemonicai` and to Docker Hub (`mnemonicai`),
  tagged per branch and release.
- Self-hosting ops (systemd service, zero-downtime model swaps via
  `mn_swap_model.sh`, backups, adapter version management) are covered in
  [`SYSTEM-CHEATSHEET.md`](SYSTEM-CHEATSHEET.md); the cloud/SaaS layer lives in
  [`cloud/`](cloud/README.md).

---

## Benchmarks

> **Results landing — see `benchmark-results/`.** Aria-14B and Aria2 evaluation
> runs (EleutherAI lm-evaluation-harness) are in progress right now; scores will
> be published here verbatim when they finish, alongside the exact harness
> settings so anyone can reproduce them. We don't publish numbers we haven't
> measured.

---

## Built in the open

MnemonicAI is built and operated by one person — **Sergio Williams (Surge)** —
working alongside AI agents, including Aria herself and AI coding assistants.
We think users deserve to know how the thing they're paying for is made:

- The base model is open (Qwen3-14B, Apache-2.0) and we credit it plainly.
- Every third-party model, library, dataset, and font is listed with its
  license in [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).
- The live site's About / How-it's-built / Licenses pages mirror this honesty,
  including what is and isn't built yet (e.g. at-rest encryption, above).

---

## Licensing

© 2026 Sergio Williams (Surge) · MnemonicAI & Aria are trademarks of Sergio
Williams · Code: source-available non-commercial · Model & content: CC BY-NC 4.0

- **Source code** — source-available for personal, educational, and research
  (non-commercial) use; commercial use requires a separate license. See
  [`LICENSE`](LICENSE).
- **Aria model weights & original content** (docs, site copy, media) —
  **CC BY-NC 4.0**. See [`LICENSE-CONTENT`](LICENSE-CONTENT). Aria derives from
  open Qwen-family weights (Apache-2.0), whose terms continue to apply upstream.
- **Third-party components** — [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).
- **Background music** — licensed via Artlist; per-track credits in
  [`assets/music/MUSIC-LICENSES.md`](assets/music/MUSIC-LICENSES.md).

Commercial licensing / questions: support@mnemonicai.org
