"""MnemonicAI Cloud gateway — the front door for the SaaS.

Serves the web app, authenticates every API call to a tenant, routes chat to
the tenant's ISOLATED memory instance, mounts Stripe billing, and accepts
codebase uploads (quota-enforced + virus-scanned). Dependency-free stdlib.

Routes:
  GET  /                       -> web app
  POST /api/signup             -> create tenant (+ Stripe checkout if paid)
  POST /api/auth               -> validate key -> tenant profile
  POST /api/chat               -> tenant-scoped chat (relays to their instance)
  GET  /api/monitor            -> tenant-scoped brain state (SSE)
  POST /api/upload             -> quota-checked, virus-scanned file intake
  POST /api/billing/portal     -> Stripe customer portal link
  POST /api/stripe-webhook     -> subscription lifecycle -> tier flip
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tenants import TenantStore, TIERS
from billing import Stripe, tier_from_event

ROOT = os.path.dirname(__file__)
WEBAPP = os.path.join(ROOT, "webapp")
# raw LLM engine (token generation only, no memory). The gateway layers each
# tenant's OWN memory on top via TenantBrains, so the shared engine never
# holds or mixes any user's memories.
RAW_ENGINE = os.environ.get("MNEM_RAW_ENGINE", "http://127.0.0.1:8402/v1")
EMBED_URL = os.environ.get("MNEM_EMBED_URL", "http://127.0.0.1:8404/v1")

store = TenantStore(root=os.environ.get("MNEM_TENANTS", "/workspace/tenants"))
stripe = Stripe()

# per-tenant isolated memory (lazy so the module imports even without the
# mnemonicai package present, e.g. in a pure-billing deployment)
_brains = None
def brains():
    global _brains
    if _brains is None:
        from tenant_memory import TenantBrains
        emb = None
        try:
            from mnemonicai.embeddings import (OpenAICompatibleEmbedder,
                                               HashingEmbedder, ResilientEmbedder)
            emb = ResilientEmbedder(
                OpenAICompatibleEmbedder(EMBED_URL, "Qwen3-Embedding"),
                HashingEmbedder())
        except Exception:
            pass
        _brains = TenantBrains(embedder=emb)
    return _brains


def clamscan(path: str) -> tuple[bool, str]:
    """Scan a file with ClamAV. Returns (clean, detail). Fails CLOSED —
    if the scanner is unavailable the upload is rejected, never trusted."""
    try:
        # --fdpass: the gateway (which owns the temp file) hands clamd the open
        # descriptor, so clamd needn't have read perms on our 0600 temp file
        r = subprocess.run(["clamdscan", "--fdpass", "--no-summary", path],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return True, "clean"
        if r.returncode == 1:
            return False, r.stdout.strip() or "malware detected"
        return False, f"scanner error: {r.stderr.strip()[:120]}"
    except FileNotFoundError:
        return False, "virus scanner unavailable — upload rejected for safety"
    except Exception as e:
        return False, f"scan failed: {str(e)[:120]}"


class Handler(BaseHTTPRequestHandler):
    server_version = "MnemonicAICloud"

    # ---- helpers ----
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self):
        try:
            return json.loads(self._body() or b"{}")
        except Exception:
            return {}

    def _tenant(self):
        """The ONLY binding of a request to a tenant. No key -> None."""
        key = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        return store.authenticate(key)

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/monitor":
            return self._monitor()
        # static web app
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        fp = os.path.normpath(os.path.join(WEBAPP, rel))
        if not fp.startswith(WEBAPP) or not os.path.isfile(fp):
            fp = os.path.join(WEBAPP, "index.html")
        ctype = "text/html" if fp.endswith(".html") else "application/octet-stream"
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/signup":      return self._signup()
            if path == "/api/auth":        return self._auth()
            if path == "/api/chat":        return self._chat()
            if path == "/api/upload":      return self._upload()
            if path == "/api/billing/portal": return self._portal()
            if path == "/api/stripe-webhook": return self._webhook()
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)[:200]})

    # ---- endpoints ----
    def _signup(self):
        d = self._json()
        email = d.get("email", "").strip()
        tier = d.get("tier", "free")
        if tier not in TIERS:
            return self._send(400, {"error": "unknown tier"})
        raw, t = store.create(email=email, tier="free")  # start free until paid
        resp = {"tenant_id": t.tenant_id, "api_key": raw, "tier": t.tier}
        if tier in ("starter", "pro"):
            price = stripe.env.get(f"STRIPE_PRICE_{tier.upper()}", "")
            base = f"https://{self.headers.get('Host','www.mnemonicai.org')}"
            sess = stripe.checkout(tier, price, email, t.tenant_id, base)
            resp["checkout_url"] = sess.get("url", "")
        return self._send(200, resp)

    def _auth(self):
        t = store.authenticate(self._json().get("key", ""))
        if not t:
            return self._send(401, {"error": "invalid key"})
        return self._send(200, {"tenant_id": t.tenant_id, "tier": t.tier,
                                "name": t.display_name,
                                "quota_gb": t.quota_bytes // 1024**3,
                                "used_bytes": store.usage_bytes(t),
                                "dedicated": t.dedicated})

    def _chat(self):
        t = self._tenant()
        if not t:
            return self._send(401, {"error": "unauthorized"})
        d = self._json()
        db = store.memory_db(t)
        # layer THIS tenant's private memory onto the prompt (isolated brain)
        msgs = brains().augment(t.tenant_id, db, d.get("messages", []))
        payload = json.dumps({
            "model": "Aerith", "messages": msgs,
            "max_tokens": min(int(d.get("max_tokens", 512)), 2048),
        }).encode()
        req = urllib.request.Request(f"{RAW_ENGINE}/chat/completions",
                                    data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "MnemonicAI-Cloud/1.0")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            reply = out.get("choices", [{}])[0].get("message", {}).get("content", "")
            brains().remember_reply(t.tenant_id, db, reply)  # perceive into their brain
            return self._send(200, out, "application/json")
        except Exception as e:
            return self._send(502, {"error": f"engine error: {str(e)[:120]}"})

    def _monitor(self):
        t = self._tenant()
        if not t:
            return self._send(401, {"error": "unauthorized"})
        # tenant-scoped brain stats only — computed from THEIR brain, so no
        # other tenant's memories can ever appear here
        try:
            s = brains().stats(t.tenant_id, store.memory_db(t))
            return self._send(200, s)
        except Exception:
            return self._send(200, {"memories": [], "note": "instance warming up"})

    def _upload(self):
        t = self._tenant()
        if not t:
            return self._send(401, {"error": "unauthorized"})
        raw = self._body()
        if not store.quota_ok(t, len(raw)):
            return self._send(413, {"error": "storage quota exceeded",
                                    "quota_gb": t.quota_bytes // 1024**3})
        fname = os.path.basename(self.headers.get("X-Filename", "upload.bin"))
        # scan in a temp file BEFORE it lands in the tenant's space
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        clean, detail = clamscan(tmp_path)
        if not clean:
            os.unlink(tmp_path)
            return self._send(422, {"error": "file rejected", "reason": detail})
        dest = os.path.join(store.uploads_dir(t), fname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(tmp_path, dest)  # /tmp and /workspace are different fs
        return self._send(200, {"stored": fname, "scan": "clean",
                                "used_bytes": store.usage_bytes(t)})

    def _portal(self):
        t = self._tenant()
        if not t or not t.stripe_customer:
            return self._send(400, {"error": "no billing account"})
        base = f"https://{self.headers.get('Host','www.mnemonicai.org')}"
        sess = stripe.portal(t.stripe_customer, base)
        return self._send(200, {"url": sess.get("url", "")})

    def _webhook(self):
        payload = self._body()
        sig = self.headers.get("Stripe-Signature", "")
        event = stripe.verify_webhook(payload, sig)
        if event is None:
            return self._send(400, {"error": "signature verification failed"})
        # persist stripe customer id on first checkout, then flip tier
        obj = event.get("data", {}).get("object", {})
        mapped = tier_from_event(event)
        if mapped:
            tenant_id, tier = mapped
            store.set_tier(tenant_id, tier)
            cust = obj.get("customer")
            if cust:
                for tt in store._by_hash.values():
                    if tt.tenant_id == tenant_id:
                        tt.stripe_customer = cust
                store._save()
        return self._send(200, {"received": True})

    def log_message(self, *a):
        pass


def main(port=8700):
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[cloud] gateway on :{port} (stripe {stripe.mode} mode, "
          f"engine {RAW_ENGINE})")
    srv.serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8700)
