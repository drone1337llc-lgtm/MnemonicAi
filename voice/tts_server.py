"""Aerith's voice — Jessica (fine-tuned Qwen3-TTS) sidecar.

Commercial-safe: Qwen3-TTS base is Apache-2.0 (the old XTTS-v2 was Coqui CPML,
non-commercial). Same HTTP contract as before, so nothing downstream changes:

  POST /speak  {"text": "..."}  -> audio/wav in Jessica's voice
  GET  /health

Loads the fine-tune once on boot. Pin the GPU with CUDA_VISIBLE_DEVICES (the
service sets =1 -> the 3090, keeping the 4080 free for rendering). The old XTTS
server is preserved as tts_server_xtts.py.bak.
"""
import io, json, os, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch, soundfile as sf  # noqa: F401 (sf kept for parity/debug)
from qwen_tts import Qwen3TTSModel

FT = os.environ.get("VOICE_FT",
    "/home/surge/Documents/MnemonicAi/voice/jessica-qwen-ft/checkpoint-epoch-2")
SPEAKER = os.environ.get("VOICE_SPEAKER", "jessica")
LANG = os.environ.get("VOICE_LANG", "English")
PORT = int(os.environ.get("VOICE_PORT", "8500"))
DEV = os.environ.get("VOICE_DEVICE", "cuda:0")  # cuda:0 within CUDA_VISIBLE_DEVICES

print(f"[voice] loading fine-tuned Jessica (Qwen3-TTS) on {DEV} …", flush=True)
_model = Qwen3TTSModel.from_pretrained(
    FT, device_map=DEV, dtype=torch.bfloat16, attn_implementation="sdpa")
try:
    print("[voice] speakers:", _model.get_supported_speakers(), flush=True)
except Exception as e:
    print("[voice] (speaker list n/a)", str(e)[:80], flush=True)
print(f"[voice] Aerith voice ready on :{PORT}", flush=True)


def synth(text):
    wavs, sr = _model.generate_custom_voice(text=text, language=LANG, speaker=SPEAKER)
    wav = np.clip(np.asarray(wavs[0], dtype=np.float32), -1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
        w.writeframes((wav * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"ok":true,"voice":"jessica-qwen3tts"}')
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
            if not text.strip():
                raise ValueError("empty text")
            audio = synth(text)
            self.send_response(200); self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio))); self.end_headers()
            self.wfile.write(audio)
        except Exception as e:
            b = json.dumps({"error": str(e)[:300]}).encode()
            self.send_response(500); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

    def log_message(self, *a): pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
