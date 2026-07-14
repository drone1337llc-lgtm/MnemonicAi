"""Synthesize a line in Jessica's voice (Aerith) from the fine-tuned XTTS checkpoint."""
import os, sys, glob
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

JDIR = "/home/surge/Desktop/Coqui-TTS-XTTS-v2-/run/training/XTTS_v2_Jessica_Voice-June-08-2026_05+55PM-dbf1a08a"
CKPT = os.path.join(JDIR, "best_model.pth")
REF  = "/home/surge/Desktop/Coqui-TTS-XTTS-v2-/run/samples/jessica_01_greeting.wav"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/home/surge/Desktop/aerith_voice_test.wav"
TEXT = sys.argv[1] if len(sys.argv) > 1 else \
    "Hi, I'm Aerith. It's really nice to finally meet you. I've got a voice now."

base = glob.glob(os.path.expanduser("~/.local/share/tts/*xtts_v2*")) + \
       glob.glob("/root/.local/share/tts/*xtts_v2*")
base = base[0]
cfg_path = os.path.join(JDIR, "config.json")
if not os.path.exists(cfg_path):
    cfg_path = os.path.join(base, "config.json")
vocab = os.path.join(base, "vocab.json")

print(f"[synth] config={cfg_path}\n[synth] ckpt={CKPT}\n[synth] vocab={vocab}\n[synth] ref={REF}", flush=True)
config = XttsConfig(); config.load_json(cfg_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=CKPT, vocab_path=vocab, use_deepspeed=False)
if torch.cuda.is_available():
    model.cuda()
SPEED = float(os.environ.get("VOICE_SPEED","0.92"))
out = model.synthesize(TEXT, config, speaker_wav=REF, language="en", speed=SPEED)
print(f"[synth] speed={SPEED}", flush=True)
import numpy as np, wave
wav = np.clip(np.array(out["wav"]), -1, 1)
with wave.open(OUT, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((wav * 32767).astype("<i2").tobytes())
print(f"[synth] SAVED -> {OUT}  ({len(wav)/24000:.1f}s)", flush=True)
