# Aerith / MnemonicAI — Master Architecture Roadmap

Goal: Aerith as a jack-of-all-trades system — while her weights stay small
enough for local hardware (RTX 3090 + 4080S). The design rule that makes this
possible: **the 12B core owns language, reasoning, identity, and memory;
every other modality is a sidecar organ the core orchestrates.** Nothing is
lost when one organ is off; each can be upgraded independently.

## Already live (do not regress)
| Capability | Where |
|---|---|
| Self-scaffolding + persistent memory | MnemonicAI core (BrainMemory, salience gate) |
| Self memory-training (sleep bakes → LoRA → weights) | trainer.py + hotswap blue/green |
| Semantic embedding retrieval | Qwen3-Embedding-0.6B sidecar :8404 (v2026-07-12) |
| Tool use / agents | OpenAI-compatible API :8400, low agent temperature |
| Autonomous operation | OpenClaw / Hermes / agentic programs on fixed model paths |
| Max token window | engine_ctx 131072 (131k), unified KV, q8 cache |
| Automatic context caching | llama.cpp cache-reuse + LCP slot reuse (already on) |
| Identity / humanizer personality | v4+ identity training + sanitized template |
| Fast decode | hybrid linear attention ~120 tok/s (12B v5: retest) |

## In flight (v5, this week)
- Qwen3.6-27B distilled dataset → depth-upscaled 12B heal → layer-prune
  ("sparsify") → benchmark-gated deploy. Reasoning/knowledge uplift.

## Phase 1 — Test-time intelligence (server-side, no new models)
- **Self-correction / reflection**: optional bridge mode — draft → self-critique
  → revise before replying (config: `reflect_mode`, per-request override).
- **Test-time scaling**: best-of-N with self-consistency vote for hard prompts
  (config: `ttc_n`), triggered by task-difficulty heuristic or client flag.
- **Advanced logic**: route "hard" prompts into an explicit structured-reasoning
  system prompt (the base can still think when asked — v4/v5 just stopped the
  compulsive scaffold).

## Phase 2 — Ears and voice (speech-to-speech)
- STT sidecar: whisper.cpp server (small.en or large-v3-turbo, CPU/GPU0).
- TTS sidecar: Kokoro-82M (Apache) — warm, fast, local.
- Bridge endpoint `/v1/audio/chat`: audio in → Aerith → audio out; monitor GUI
  gets a mic button. Latency budget ~1.5s round trip.

## Phase 3 — Eyes (vision)
- Qwen3.5-VL sidecar (9B, Apache) via llama.cpp mmproj on GPU0; bridge routes
  image-bearing requests: VL describes/reads → Aerith reasons and answers
  (keeps Aerith's memory + identity in the loop, zero retraining).
- Later: graft a vision tower onto Aerith herself during a training cycle
  (her Qwen3.5 lineage is natively multimodal — preprocessor configs exist).

## Phase 4 — Hands (computer use + search)
- **High-accuracy search**: web search tool (SearXNG local or API) with
  embedding-sidecar reranking; results injected as context with citations;
  memory system stores what she learns.
- **Computer use**: screenshot → VL sidecar → Aerith plans → executes via
  existing tool-calling. Gate destructive actions behind confirmation.

## Phase 5 — Imagination (image/video generation)
- ComfyUI or sd-server sidecar (FLUX-schnell / LTX-Video class, licenses
  permitting) on the 4080S; Aerith writes the prompts, gallery in monitor GUI.
- These are generators she *uses*, not weights she carries.

## Sequencing rationale
Phase 1 is pure software on the existing engine (days). Phases 2-3 are mature
local sidecars (each a weekend). Phase 4 composes 1+3. Phase 5 is bolt-on.
Every phase ships independently; none blocks the sleep-training loop, and the
strict `Aerith` naming contract never changes.
