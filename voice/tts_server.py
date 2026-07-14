"""Aerith's voice — Jessica (fine-tuned XTTS-v2) TTS sidecar.

  POST /speak  {"text": "..."}  -> audio/wav in Jessica's voice (speed 0.92)
  GET  /health

Loads once on boot (5.6GB checkpoint on GPU). The MnemonicAI bridge/gateway
POSTs Aerith's text reply here to get spoken audio back.
"""
import io, json, os, glob, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

JDIR = os.environ.get("VOICE_DIR",
    "/home/surge/Desktop/Coqui-TTS-XTTS-v2-/run/training/XTTS_v2_Jessica_Voice-June-08-2026_05+55PM-dbf1a08a")
CKPT = os.path.join(JDIR, "best_model.pth")
REF = os.environ.get("VOICE_REF",
    "/home/surge/Desktop/Coqui-TTS-XTTS-v2-/run/samples/jessica_01_greeting.wav")
SPEED = float(os.environ.get("VOICE_SPEED", "0.92"))
PORT = int(os.environ.get("VOICE_PORT", "8500"))
_base = (glob.glob(os.path.expanduser("~/.local/share/tts/*xtts_v2*")) +
         glob.glob("/root/.local/share/tts/*xtts_v2*"))[0]

print(f"[voice] loading Jessica XTTS (speed {SPEED}) …", flush=True)
_cfg = XttsConfig(); _cfg.load_json(os.path.join(_base, "config.json"))
_model = Xtts.init_from_config(_cfg)
_model.load_checkpoint(_cfg, checkpoint_path=CKPT,
                       vocab_path=os.path.join(_base, "vocab.json"), use_deepspeed=False)
if torch.cuda.is_available():
    _model.cuda()
# precompute speaker latents once (faster per-request)
_gpt_lat, _spk_emb = _model.get_conditioning_latents(audio_path=[REF])
print(f"[voice] Aerith voice ready on :{PORT}", flush=True)

def synth(text):
    out = _model.inference(text, "en", _gpt_lat, _spk_emb, speed=SPEED)
    wav = np.clip(np.array(out["wav"]), -1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((wav * 32767).astype("<i2").tobytes())
    return buf.getvalue()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
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
