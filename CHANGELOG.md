# Changelog

All notable changes to **MnemonicAi**. Versions follow semver-ish product milestones.

## v2.2.0 — "Sight & Sound" (2026-07-02)

### Added
- **⧖ Session timeline view** — third monitor view: a scrubbable horizontal history of the
  session. Memory births as dots on kind lanes, reinforcements as ticks along each
  lifespan line, forgettings as ✕ marks, and LoRA bakes as gold ✦ markers with adapter
  versions. Scroll zooms time, drag pans, ⤢ snaps back to live-follow; hover for details,
  click a living memory to inspect it.
- **Sound** — synthesized sound effects (no audio files needed) for gate admit/reject,
  recall, reinforcement, eviction, sleep consolidation, bake start/done/rollback, view
  transitions, and clicks; plus a licensed **background soundtrack** (Artlist) with an
  ambient default playlist (*That's What It Was*, *Desire*, *Silent Transmission*).
  🔊 master mute button (also `m`), track name in the footer.
- **🎵 Music-reactive memories** — an equalizer adapted to the brain: each memory is
  assigned a frequency band and pulses (size + glow) to the music in every view; region
  cores swell with overall energy. Toggling reactive switches to the energetic playlist
  (Pulse, Stunts Cheer, get this work, finallyfree, Going Back to the Old School,
  Groove It Forward, Make Me Sick).
- **⛶ Zen mode** — hide the entire UI (also `h`) for a clean look at the brain/graph/
  timeline; floating mini-bar keeps mute + exit within reach.
- **Auto consolidation cards** — every successful bake writes a cosmic SVG "memory card"
  (adapter version, loss, eval delta, the memories baked) to `mnemonicai_data/cards/`,
  announced in the monitor event log.
- Static asset serving (`/assets/…`) in the pure-stdlib server, with path-traversal
  protection.
- GitHub-ready README, this CHANGELOG, `.gitignore`, and `assets/music/MUSIC-LICENSES.md`.

### Changed
- View toggle now cycles **Brain → Graph → Timeline**; zoom controls adapt per view.
- Version reported consistently as 2.2.0.

### Fixed
- Timeline records persist for the whole session even after a memory is forgotten.

## v2.1.0 — "Constellations" (2026-07-02)

- **Obsidian-style graph view**: force-directed memory graph (nodes = memories, edges =
  Hebbian links) with hover-highlighting, dragging, and per-kind colors.
- **Physical view transitions**: memories explode out of the brain into the graph and
  magnetically regroup onto their lobes on return (with shockwaves/ripples).
- **Click-to-inspect** overlay in both views: text, strength, recalls, connections,
  pin/delete, **PNG graph-card** and **JSON** export.
- **Density-aware dots** (small, capped per region, spread with crowding) and zoom/pan.
- **Memories drawer** with search, 📌 pin (never pruned), delete.
- **Bottom bake bar** replaces the old scrolling gold band — invisible until a bake runs.
- Anatomical brain rendered from a real reference image (clean-edge extraction, pre-baked
  glow), cosmic Milky-Way theme.
- Engine: pinned memories protected from pruning; state payload exposes memories + link
  graph; `/api/memory/pin`; HF-cache model-dir auto-resolution (`blobs/refs/snapshots`).
- Renamed project **Mnemosyne → MnemonicAi** throughout (package `mnemonicai`, CLI
  `mnemonicai serve`, env vars `MNEMONICAI_*`, data dir `mnemonicai_data/`).

## v2.0.0 — "Two-Speed Memory" (2026-07-02)

- **Self-hosted inference + continual learning**: Transformers + PEFT **QLoRA** backend
  (NVIDIA) sharing one in-process adapter between inference and sleep-training — baked
  memories apply instantly and persist on disk; dependency-free **mock backend** runs the
  whole product without a GPU.
- **OpenAI-compatible server** (pure stdlib): `/v1/chat/completions` (+streaming),
  `/v1/models`, SSE `/events`, admin API (perceive/recall/sleep/train/reset/memories).
- **Sleep-trainer guardrails**: base-capability replay buffer, eval-based rollback,
  adapter versioning with pruning.
- **Live brain monitor**: real-time visualization of the memory engine over SSE with a
  built-in simulated fallback; control bar (inject/recall/sleep/bake/reset).
- One-file `install.py` / `start.py`, pip entry point, Docker + compose.

## v1.0.0 — "The Engine" (2026-07-02)

- Brain-inspired memory engine (zero dependencies): sensory buffer → salience gate →
  working memory (7±2, activation decay) → sleep consolidation → episodic/semantic/
  procedural stores; Ebbinghaus decay, spacing-effect reinforcement, Hebbian links,
  reconstructive retrieval, pruning; SQLite persistence (long-term only, like a brain).
- Framework-agnostic adapter hooks + LM Studio preset; offline demo, chat REPL, and the
  standalone concept simulator; 11 unit tests.
