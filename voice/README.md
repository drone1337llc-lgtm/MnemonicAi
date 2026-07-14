# Aerith's voice — Jessica (XTTS-v2 TTS sidecar)

Gives Aerith spoken output in Surge's trained "Jessica" voice. This is the
speech-OUTPUT half the omni model lacks (omni hears + sees; this speaks).

## Service
- `aerith-voice.service` (systemd, CudaCuda) runs `tts_server.py` on **:8500**,
  GPU-loaded, speed 0.92 (tuned with Surge).
- `POST /speak {"text":"..."}` -> `audio/wav` (24kHz) in Jessica's voice.
- `GET /health`.

## Model (not in git — 5.6GB)
- Checkpoint: `.../Coqui-TTS-XTTS-v2-/run/training/XTTS_v2_Jessica_Voice-*/best_model.pth`
- Base vocab/config: `~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/`
- Reference clip: `.../run/samples/jessica_01_greeting.wav`
- ⚠️ Currently loaded from the **Desktop** path (fragile). Move the Coqui folder
  to a stable location and set `VOICE_DIR` / `VOICE_REF` in the service if you
  clear the Desktop.

## Wire into Aerith (text reply -> speech)
Any client with Aerith's text reply POSTs it to `/speak`:
```bash
REPLY=$(curl -s .../v1/chat/completions ... | jq -r .choices[0].message.content)
curl -s -X POST http://127.0.0.1:8500/speak -H "Content-Type: application/json" \
  -d "{\"text\":\"$REPLY\"}" -o reply.wav
```
For the brain-monitor GUI: add a "speak" button that POSTs the last reply here
and plays the returned wav. For the omni pod: run this sidecar and have the
gateway offer `/api/speak`.

## Tuning knobs (env on the service)
- `VOICE_SPEED` (0.92 now; lower = slower)
- `VOICE_REF` (swap reference clip to shift timbre)
For finer pace control than XTTS allows, post-process with a pitch-preserving
time-stretch (rubberband/librosa).

## License note (commercial)
The fine-tune derives from Coqui XTTS-v2 (CPML — non-commercial). For the paid
SaaS, confirm licensing or use a commercially-licensed base before shipping voice.
