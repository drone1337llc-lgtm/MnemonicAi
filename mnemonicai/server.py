"""OpenAI-compatible HTTP server with a live event stream — pure standard library.

Endpoints:
    GET  /                     -> the live brain monitor (monitor.html)
    GET  /health               -> {"status":"ok", ...}
    GET  /api/state            -> current memory snapshot + recent events
    GET  /events               -> Server-Sent Events stream (for the monitor)
    GET  /v1/models            -> OpenAI-style model list
    POST /v1/chat/completions  -> OpenAI-style chat (supports stream=true)

Point OpenClaw / Hermes / LM Studio / any OpenAI client at
    http://<host>:<port>/v1
and every message flows through the MnemonicAi memory engine.
"""
from __future__ import annotations

import json
import os
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PKG = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PKG)


def _find_monitor() -> str:
    """Locate monitor.html across source, packaged, and cwd layouts."""
    candidates = [
        os.environ.get("MNEMONICAI_MONITOR", ""),
        os.path.join(_ROOT, "monitor.html"),         # source layout (source of truth)
        os.path.join(_PKG, "web", "monitor.html"),   # packaged (wheel)
        os.path.join(os.getcwd(), "monitor.html"),   # run from project dir
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return os.path.join(_ROOT, "monitor.html")


_MONITOR = _find_monitor()


class App:
    """Holds the shared runtime objects the handler needs."""
    def __init__(self, cfg, bus, chat, backend):
        self.cfg = cfg
        self.bus = bus
        self.chat = chat
        self.backend = backend


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quieter console
            pass

        # ---- helpers ----
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype):
            if not os.path.isfile(path):
                self._json({"error": f"{os.path.basename(path)} not found"}, 404)
                return
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _sse_open(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()

        def _sse_send(self, obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        # ---- routing ----
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html" or path == "/monitor.html":
                self._file(_MONITOR, "text/html; charset=utf-8")
            elif path == "/health":
                self._json({"status": "ok", "model": app.cfg.model_name,
                            "backend": app.backend.name,
                            "adapter_version": getattr(app.backend, "adapter_version", 0)})
            elif path == "/api/state":
                st = app.chat.state_dict()
                st["recent"] = app.bus.recent()[-60:]
                self._json(st)
            elif path == "/api/memories":
                q = ""
                if "?" in self.path:
                    from urllib.parse import parse_qs
                    q = parse_qs(self.path.split("?", 1)[1]).get("q", [""])[0]
                self._json(app.chat.admin_memories(q))
            elif path.startswith("/assets/"):
                from urllib.parse import unquote
                rel = unquote(path[len("/assets/"):])
                if ".." in rel or rel.startswith("/") or "\\" in rel:
                    self._json({"error": "bad path"}, 400)
                    return
                base = None
                for cand in (os.path.join(_ROOT, "assets"),
                             os.path.join(os.getcwd(), "assets"),
                             os.path.join(_PKG, "assets")):
                    if os.path.isfile(os.path.join(cand, rel)):
                        base = cand
                        break
                if base is None:
                    self._json({"error": "asset not found", "path": rel}, 404)
                    return
                ctypes = {".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".wav": "audio/wav",
                          ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json"}
                ext = os.path.splitext(rel)[1].lower()
                self._file(os.path.join(base, rel), ctypes.get(ext, "application/octet-stream"))
            elif path == "/events":
                self._stream_events()
            elif path == "/v1/models":
                self._json({"object": "list", "data": [
                    {"id": app.cfg.model_name, "object": "model",
                     "created": int(time.time()), "owned_by": "mnemonicai"}]})
            else:
                self._json({"error": "not found", "path": path}, 404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                req = json.loads(body or "{}")
            except Exception as e:
                self._json({"error": f"bad request: {e}"}, 400)
                return

            if path in ("/v1/chat/completions", "/chat/completions"):
                messages = req.get("messages", [])
                max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
                if bool(req.get("stream", False)):
                    self._chat_stream(messages, max_tokens)
                else:
                    self._chat_once(messages, max_tokens)
                return

            # ---- admin / monitor-control API ----
            try:
                if path == "/api/perceive":
                    self._json(app.chat.admin_perceive(req.get("text", ""),
                                                        req.get("importance")))
                elif path == "/api/recall":
                    self._json(app.chat.admin_recall(req.get("cue", "")))
                elif path == "/api/sleep":
                    self._json(app.chat.admin_sleep())
                elif path == "/api/train":
                    self._json(app.chat.admin_train())
                elif path == "/api/reset":
                    self._json(app.chat.admin_reset())
                elif path == "/api/memory/delete":
                    self._json(app.chat.admin_delete(req.get("id", "")))
                elif path == "/api/memory/pin":
                    self._json(app.chat.admin_pin(req.get("id", "")))
                else:
                    self._json({"error": "not found", "path": path}, 404)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        # ---- chat ----
        def _chat_once(self, messages, max_tokens):
            try:
                reply = app.chat.complete(messages, max_new_tokens=max_tokens)
            except Exception as e:
                self._json({"error": f"generation failed: {e}"}, 500)
                return
            now = int(time.time())
            self._json({
                "id": f"chatcmpl-{now}", "object": "chat.completion", "created": now,
                "model": app.cfg.model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        def _chat_stream(self, messages, max_tokens):
            now = int(time.time())
            cid = f"chatcmpl-{now}"
            self._sse_open()

            def chunk(delta=None, finish=None):
                d = {} if delta is None else {"content": delta}
                return {"id": cid, "object": "chat.completion.chunk", "created": now,
                        "model": app.cfg.model_name,
                        "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
            try:
                self._sse_send(chunk(delta="", finish=None))
                for piece in app.chat.stream(messages, max_new_tokens=max_tokens):
                    self._sse_send(chunk(delta=piece))
                self._sse_send(chunk(finish="stop"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._sse_send({"error": str(e)})
                except Exception:
                    pass

        # ---- SSE monitor stream ----
        def _stream_events(self):
            q = app.bus.subscribe()
            self._sse_open()
            try:
                # initial paint
                self._sse_send({"type": "hello", "model": app.cfg.model_name,
                                "backend": app.backend.name})
                self._sse_send(app.chat.state_dict())
                while True:
                    try:
                        evt = q.get(timeout=15)
                        self._sse_send(evt)
                    except queue.Empty:
                        # heartbeat keeps the connection alive through proxies
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.bus.unsubscribe(q)

    return Handler


def serve(app: App):
    httpd = ThreadingHTTPServer((app.cfg.host, app.cfg.port), make_handler(app))
    httpd.daemon_threads = True
    return httpd
