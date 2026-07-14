"""Clone Jessica's voice with Qwen3-TTS (Apache-2.0) — commercial-safe replacement
for XTTS. Zero-shot clone from one reference clip + its transcript."""
import sys, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

REF_WAV  = "/home/surge/Desktop/Coqui-TTS-XTTS-v2-/data/jessica_voice/wavs/jessica_0002.wav"
REF_TEXT = "I've saved your work and backed everything up, so you're all set."
TEXT = sys.argv[1] if len(sys.argv) > 1 else \
    "Hi, I'm Aerith. It's really nice to finally meet you. Sergio gave me a voice, and I think it suits me."
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/home/surge/Desktop/aerith_voice_qwen.wav"

print("[qwen-tts] loading Qwen3-TTS-12Hz-1.7B-Base …", flush=True)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
print("[qwen-tts] cloning Jessica …", flush=True)
wavs, sr = model.generate_voice_clone(
    text=TEXT, language="English", ref_audio=REF_WAV, ref_text=REF_TEXT)
sf.write(OUT, wavs[0], sr)
print(f"[qwen-tts] SAVED -> {OUT} ({len(wavs[0])/sr:.1f}s @ {sr}Hz)", flush=True)
