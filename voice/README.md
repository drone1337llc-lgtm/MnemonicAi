# Aerith's voice — Jessica (Qwen3-TTS sidecar)

Gives Aerith spoken output in Surge's trained "Jessica" voice. This is the
speech-OUTPUT half the omni model lacks (omni hears + sees; this speaks).

**Commercial-safe:** the voice base is **Qwen3-TTS-12Hz-1.7B (Apache-2.0)**,
fine-tuned on Jessica. It replaced the old Coqui XTTS-v2 build (CPML —
non-commercial) as of 2026-07-14. The previous XTTS server is preserved at
`tts_server_xtts.py.bak`.

## Service
- `aerith-voice.service` (systemd, CudaCuda) runs `tts_server.py` on **:8500**,
  pinned to the 3090 via `CUDA_VISIBLE_DEVICES=1` (keeps the 4080 free).
- Runs under the `aerith_qwentts_venv` (has `qwen_tts`; the old `aerith_voice_venv`
  had Coqui TTS).
- `POST /speak {"text":"..."}` -> `audio/wav` (24kHz) in Jessica's voice.
- `GET /health` -> `{"ok":true,"voice":"jessica-qwen3tts"}`.

## Model (not in git)
- Fine-tune checkpoint: `voice/jessica-qwen-ft/checkpoint-epoch-2` (~3.4GB, bf16).
- Speaker id registered in the checkpoint: `jessica`.
- Generation API (see `gen_finetuned.py`):
  `Qwen3TTSModel.from_pretrained(FT, ...).generate_custom_voice(text=..., language="English", speaker="jessica")`.

## Wire into Aerith (text reply -> speech)
Any client with Aerith's text reply POSTs it to `/speak`:
```bash
REPLY=$(curl -s .../v1/chat/completions ... | jq -r .choices[0].message.content)
curl -s -X POST http://127.0.0.1:8500/speak -H "Content-Type: application/json" \
  -d "{\"text\":\"$REPLY\"}" -o reply.wav
```
For the brain-monitor / web GUI: add a "speak" button that POSTs the last reply
here and plays the returned wav. For the omni pod: run this sidecar and have the
gateway offer `/api/speak`.

## Tuning knobs (env on the service)
- `VOICE_DEVICE` (default `cuda:0` within `CUDA_VISIBLE_DEVICES`) — which GPU.
- `VOICE_FT` — swap the fine-tune checkpoint.
- `VOICE_SPEAKER` / `VOICE_LANG` — speaker id / language.
- `VOICE_PORT` — default 8500.
The Qwen output pace was accepted as-is (no speed hack needed, unlike XTTS which
ran at 0.92). For finer pace control, post-process with a pitch-preserving
time-stretch (rubberband/librosa).
