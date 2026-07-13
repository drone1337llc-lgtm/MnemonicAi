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


def _normalize_messages(messages):
    """Coerce OpenAI-style messages into plain {role, content:str} dicts.

    Clients like OpenClaw send `content` as a list of typed parts
    ([{"type":"text","text":...}]), tool-role messages, or null content —
    all of which break tokenizer chat templates that expect plain strings.
    """
    norm = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    text = p.get("text") or p.get("content") or ""
                    if text and p.get("type") in (None, "text", "input_text"):
                        parts.append(str(text))
            c = "\n".join(parts)
        elif c is None:
            c = ""
        if role == "tool":
            role, c = "user", f"[tool result] {c}"
        elif role not in ("system", "user", "assistant"):
            role = "user"
        norm.append({"role": role, "content": str(c)})
    return norm


def _sanitize_for_template(messages):
    """Make a raw client message list safe for the model's chat template
    without losing tool-calling structure (unlike _normalize_messages,
    which flattens everything to plain text).

    Agent clients (Hermes) append empty assistant placeholders, producing
    trailing/consecutive assistant turns that the Qwen template rejects
    ("Cannot have 2 or more assistant messages at the end of the list").
    Drop content-less assistant messages that carry no tool_calls, and
    merge any assistant turns that remain adjacent.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        role = m.get("role")
        content = m.get("content")
        has_text = bool(content.strip()) if isinstance(content, str) else (
            content is not None and not isinstance(content, str))
        has_tc = bool(m.get("tool_calls"))
        # drop empty assistant placeholders (no text, no tool call)
        if role == "assistant" and not has_text and not has_tc:
            continue
        # merge an assistant turn that would sit right after another
        if role == "assistant" and out and out[-1].get("role") == "assistant":
            prev = out[-1]
            pc, cc = prev.get("content") or "", content or ""
            if isinstance(pc, str) and isinstance(cc, str):
                prev["content"] = pc + ("\n" if pc and cc else "") + cc
            elif cc:
                prev["content"] = cc
            if m.get("tool_calls"):
                prev["tool_calls"] = (prev.get("tool_calls") or []) + m["tool_calls"]
            continue
        out.append(m)
    return out


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
            # HTTP/1.1 + no Content-Length + no chunked encoding means the
            # body MUST be delimited by connection close. With keep-alive,
            # spec-compliant clients (Node/undici = OpenClaw) wait forever
            # for the response to "end" and never surface the stream.
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

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
                messages = _normalize_messages(req.get("messages", []))
                max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
                # Clamp the requested output budget: llama-server counts
                # prompt + max_tokens against the context window and rejects
                # the whole request when the sum exceeds it. Agent clients
                # (Hermes) often ask for huge outputs (e.g. 60000), which
                # turned a 19k-token prompt into an instant 400.
                cap = getattr(app.cfg, "max_new_tokens_cap", 8192)
                if max_tokens and cap:
                    try:
                        max_tokens = min(int(max_tokens), cap)
                    except (TypeError, ValueError):
                        max_tokens = None
                if os.environ.get("MNEMONICAI_DEBUG_REQUESTS"):
                    total = sum(len(m["content"]) for m in messages)
                    print(f"[req] {self.client_address[0]} stream={bool(req.get('stream'))} "
                          f"msgs={len(messages)} chars={total} max_tokens={max_tokens} "
                          f"tools={len(req.get('tools') or [])} "
                          f"extra_keys={sorted(set(req) - {'messages', 'model', 'stream', 'max_tokens', 'temperature', 'top_p'})}",
                          flush=True)
                # Tool-calling / agent requests (Hermes): the memory pipeline
                # returns plain text and would drop `tools` and the resulting
                # `tool_calls`, so the client stalls waiting for an action.
                # Hand these straight to the engine, which speaks tools.
                if req.get("tools") and hasattr(app.backend, "proxy_chat"):
                    self._chat_tools_passthrough(req, max_tokens)
                    return
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
                elif path == "/admin/swap-base":
                    # Zero-downtime base-model switch (hybrid backend only).
                    # Body: {"gguf_path": "...", "model_path": "..."(optional),
                    #        "model_name": "..."(optional)}
                    if not hasattr(app.backend, "mgr"):
                        self._json({"error": "base swapping needs the hybrid "
                                    "backend (blue/green llama-server pair); "
                                    f"current backend is '{app.backend.name}'"},
                                   409)
                        return
                    gguf = req.get("gguf_path", "")
                    if not gguf or not os.path.isfile(gguf):
                        self._json({"error": f"gguf_path not found: '{gguf}'"},
                                   400)
                        return
                    model_path = req.get("model_path", "")
                    if model_path and not os.path.isdir(model_path):
                        self._json({"error": "model_path (HF dir for "
                                    f"sleep-training) not found: '{model_path}'"},
                                   400)
                        return
                    app.backend.mgr.swap_base(gguf)  # raises if unhealthy
                    # persist so restarts and sleep-training use the new base
                    app.cfg.gguf_path = gguf
                    if model_path:
                        app.cfg.model_path = model_path
                    if req.get("model_name"):
                        app.cfg.model_name = req["model_name"]
                    app.cfg.save()
                    self._json({"status": "swapped",
                                "gguf_path": gguf,
                                "model_path": app.cfg.model_path,
                                "adapter_version": 0,
                                "note": "memory adapter reset; sleep-training "
                                        "rebuilds it on the new base"})
                elif path == "/admin/apply-adapter":
                    # Zero-downtime install of an ALREADY-CONVERTED adapter
                    # GGUF (hybrid backend only) — for adapters trained
                    # offline (e.g. an overnight standalone run) rather than
                    # through this server's own sleep-training loop. Body:
                    # {"adapter_gguf": "...", "version": optional int}
                    if not hasattr(app.backend, "mgr"):
                        self._json({"error": "adapter swapping needs the "
                                    "hybrid backend (blue/green llama-server "
                                    f"pair); current backend is "
                                    f"'{app.backend.name}'"}, 409)
                        return
                    adapter_gguf = req.get("adapter_gguf", "")
                    if not adapter_gguf or not os.path.isfile(adapter_gguf):
                        self._json({"error": f"adapter_gguf not found: "
                                    f"'{adapter_gguf}'"}, 400)
                        return
                    version = req.get("version")
                    if version is None:
                        version = app.backend.adapter_version + 1
                    app.backend.mgr.swap_in(adapter_gguf, version)  # raises if unhealthy
                    from .backend import _write_version
                    _write_version(app.cfg.adapter_dir, version)
                    self._json({"status": "swapped", "adapter_gguf": adapter_gguf,
                                "adapter_version": version})
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

        # ---- tool-calling passthrough (agent clients) ----
        def _perceive_last_user(self, req):
            """Feed the latest user turn into memory (best-effort). We don't
            inject recall into agent context — that could derail the agent —
            but learned memory still reaches the model via the trained
            adapter."""
            try:
                for m in reversed(req.get("messages", [])):
                    if m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, list):
                            c = " ".join(p.get("text", "") for p in c
                                         if isinstance(p, dict))
                        if isinstance(c, str) and c.strip():
                            app.chat.admin_perceive(c.strip()[:4000], None)
                        break
            except Exception as e:
                print(f"[passthrough] perceive skipped: {e}", flush=True)

        def _chat_tools_passthrough(self, req, max_tokens):
            payload = dict(req)
            payload["messages"] = _sanitize_for_template(req.get("messages"))
            if max_tokens:
                payload["max_tokens"] = max_tokens
                payload.pop("max_completion_tokens", None)
            # Agent clients (Hermes) send no sampling params, so the engine
            # falls back to its high default (~0.8) — which makes this model
            # emit tool calls erratically or refuse. Pin a low temperature
            # for tool requests unless the client set one explicitly.
            if req.get("temperature") is None:
                payload["temperature"] = getattr(app.cfg, "agent_temperature", 0.2)
            if req.get("top_p") is None:
                payload["top_p"] = getattr(app.cfg, "agent_top_p", 0.9)
            stream = bool(req.get("stream", False))
            try:
                resp = app.backend.proxy_chat(payload)
            except Exception as e:
                # surface the engine's own error body if there is one
                detail = getattr(e, "read", lambda: b"")()
                msg = detail.decode("utf-8", "replace")[:500] if detail else str(e)
                self._json({"error": f"engine passthrough failed: {msg}"}, 502)
                return
            self._perceive_last_user(req)
            try:
                if stream:
                    self._sse_open()
                    for raw in resp:  # relay the engine's SSE verbatim
                        if raw:
                            self.wfile.write(raw)
                            self.wfile.flush()
                else:
                    body = resp.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self._cors()
                    self.end_headers()
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    resp.close()
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
