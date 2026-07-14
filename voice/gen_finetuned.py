import sys, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
FT="/home/surge/Documents/MnemonicAi/voice/jessica-qwen-ft/checkpoint-epoch-2"
TEXT=sys.argv[1] if len(sys.argv)>1 else "Hi, I'm Aerith. It's really nice to finally meet you. Sergio gave me a voice, and I think it suits me."
OUT=sys.argv[2] if len(sys.argv)>2 else "/home/surge/Desktop/aerith_voice_qwen_finetuned.wav"
print("[gen] loading fine-tuned Jessica (custom_voice)...", flush=True)
m=Qwen3TTSModel.from_pretrained(FT, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
try: print("[gen] speakers:", m.get_supported_speakers())
except Exception as e: print("[gen] (speakers list n/a)", str(e)[:60])
wavs,sr=m.generate_custom_voice(text=TEXT, language="English", speaker="jessica")
sf.write(OUT, wavs[0], sr)
print(f"[gen] SAVED -> {OUT} ({len(wavs[0])/sr:.1f}s)", flush=True)
