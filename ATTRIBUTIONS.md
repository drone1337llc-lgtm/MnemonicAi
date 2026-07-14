# Attributions, citations & third-party notices

MnemonicAI is built openly on the shoulders of excellent open-source and
open-weight work. This page lists everything in the project that is **not
completely original**, with its author and license, in the spirit of full
transparency about how Aria was made.

Nothing here implies endorsement by the listed projects. Where a component is
used unmodified as a service or external binary, that is noted.

Last updated: 2026-07-14.

---

## Model weights & AI models

| Component | Author | License | Role in MnemonicAI |
|---|---|---|---|
| **Qwen** (Qwen2.5 / Qwen3 family) | Alibaba Cloud (Qwen Team) | Apache-2.0 | Base lineage that Aria descends from (via merging + our own training). |
| **Qwen3-Embedding-0.6B** | Alibaba Cloud (Qwen Team) | Apache-2.0 | Semantic memory retrieval (embeddings sidecar). |
| **Qwen3-TTS (12Hz-1.7B base)** | Alibaba Cloud (Qwen Team) | Apache-2.0 | Base for Aria's spoken voice, fine-tuned on the owner's "Jessica" voice. |
| **Qwen3 (teacher)** | Alibaba Cloud (Qwen Team) | Apache-2.0 | Teacher model for the v5 distillation dataset. |

**Aria itself** is a derivative work: open Qwen-family weights, merged and then
trained (LoRA fine-tunes, continued pre-training, identity/format repair, and a
depth-upscale + heal for the v5 line) by Sergio Williams (Surge). See the
[model card / version file](models) and the Licenses section below for how the
resulting weights are licensed.

## Model tooling & frameworks

| Component | Author | License |
|---|---|---|
| **PyTorch** | Meta / PyTorch Foundation | BSD-3-Clause |
| **Hugging Face Transformers** | Hugging Face | Apache-2.0 |
| **PEFT** (LoRA) | Hugging Face | Apache-2.0 |
| **Datasets** | Hugging Face | Apache-2.0 |
| **Accelerate** | Hugging Face | Apache-2.0 |
| **safetensors** | Hugging Face | Apache-2.0 |
| **SentencePiece** | Google | Apache-2.0 |
| **bitsandbytes** | Tim Dettmers & contributors | MIT |
| **llama.cpp** | Georgi Gerganov & contributors | MIT |
| **mergekit** | Arcee AI | LGPL-3.0-only |
| **lm-evaluation-harness** | EleutherAI | MIT |

## Application & serving stack

| Component | Author | License |
|---|---|---|
| **FastAPI** | Sebastián Ramírez | MIT |
| **Starlette / Uvicorn** | Encode | BSD-3-Clause |
| **sse-starlette** | Alex Rudenko | BSD-3-Clause |
| **Pydantic** | Pydantic Services | MIT |
| **httpx** | Encode | BSD-3-Clause |
| **python-multipart** | Andrew Dunham | Apache-2.0 |
| **prometheus-client / -fastapi-instrumentator** | Prometheus / T. Volkmann | Apache-2.0 / ISC |
| **ClamAV** | Cisco Systems | GPL-2.0 (run as an external scanning daemon; not linked into our code) |

## Training data

| Source | License / terms | Use |
|---|---|---|
| **Wikipedia** | CC BY-SA (Wikimedia) | Continued pre-training corpus (mixed with the below). |
| **RedPajama** | Together Computer (component licenses vary; permissive/open) | Continued pre-training corpus. |
| **Qwen3 teacher generations** | Output of an Apache-2.0 model | v5 distillation dataset. |
| **Identity / format-repair set** | Original, authored for this project | Teaches Aria's name, creator, and answer format. |

## Voice

Aria's spoken voice ("Jessica") was **trained by the owner** on a
commercially-licensed Apache-2.0 base (Qwen3-TTS). An earlier prototype used
Coqui XTTS-v2 (CPML, non-commercial); it is **not** used in the shipped product.

## Music

All background music is licensed via **Artlist** (artlist.io) under Song
Certificates issued to the project owner. Per-track credits (title, artist) are
in [`assets/music/MUSIC-LICENSES.md`](assets/music/MUSIC-LICENSES.md). These
recordings are licensed for use **within MnemonicAI only**.

## Fonts & UI

- **Inter** typeface — Rasmus Andersson — SIL Open Font License 1.1.
- Emoji/glyphs render with the viewer's own system fonts.

## Infrastructure & services

RunPod (GPU compute), Cloudflare (tunnels / edge), Stripe (payments), GitHub +
GHCR + Docker Hub (source & image hosting), Artlist (music). These are used as
services under their respective terms; no proprietary code of theirs is
redistributed here.

---

*If you believe something is used here without proper attribution, please
contact support@mnemonicai.org and we will correct it promptly.*
