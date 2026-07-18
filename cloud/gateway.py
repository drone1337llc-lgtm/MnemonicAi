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
RAW_ENGINE_KEY = os.environ.get("MNEM_RAW_ENGINE_KEY", "")  # keyed engines (omni vLLM)
# default identity for engines that aren't identity-fine-tuned (e.g. raw omni)
ARIA_IDENTITY = os.environ.get("MNEM_IDENTITY", "")
EMBED_URL = os.environ.get("MNEM_EMBED_URL", "http://127.0.0.1:8404/v1")
# n8n automation webhooks (observe/notify only — never billing/tenant writes)
N8N_HOOKS = os.environ.get("MNEM_N8N_HOOKS", "https://hooks.mnemonicai.org/webhook")

store = TenantStore(root=os.environ.get("MNEM_TENANTS", "/workspace/tenants"))
stripe = Stripe()

# Mission Control dashboard usage feed (best-effort; billing never depends on it)
from mc_emitter import MCEmitter
_env = stripe.env
mc = MCEmitter(_env.get("MC_USAGE_URL", ""), _env.get("MC_BUSINESS_KEY", ""))


def _fire_hook(path, payload):
    """Best-effort POST to an n8n observe/notify webhook. Never blocks or breaks
    the request if n8n is unreachable."""
    if not N8N_HOOKS:
        return
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{N8N_HOOKS}/{path}", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=4)
    except Exception:
        pass

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
        if path == "/api/admin/trials":
            return self._admin_trials()
        # static web app
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        if not rel.endswith(".html") and os.path.isfile(os.path.join(WEBAPP, rel + ".html")):
            rel += ".html"  # allow extensionless routes like /tools
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
        tier = d.get("tier", "trial")
        if tier not in ("trial", "starter", "pro"):  # public signup tiers only
            return self._send(400, {"error": "unknown plan"})
        if tier == "trial":
            # no credit card — instant, time-limited taste of the product
            raw, t = store.create(email=email, tier="trial")
            _fire_hook("new-signup", {"email": email, "tier": "trial",
                                      "tenant_id": t.tenant_id,
                                      "trial_days_left": t.trial_days_left})
            return self._send(200, {"tenant_id": t.tenant_id, "api_key": raw,
                                    "tier": "trial", "trial_days_left": t.trial_days_left,
                                    "message": "Your free trial is live — no card needed."})
        # paid: create as trial-on-that-tier, Stripe collects card + 15-day free trial
        raw, t = store.create(email=email, tier=tier)
        _fire_hook("new-signup", {"email": email, "tier": tier,
                                  "tenant_id": t.tenant_id, "trial_days_left": 0})
        price = stripe.env.get(f"STRIPE_PRICE_{tier.upper()}", "")
        base = f"https://{self.headers.get('Host','www.mnemonicai.org')}"
        sess = stripe.checkout(tier, price, email, t.tenant_id, base)
        return self._send(200, {"tenant_id": t.tenant_id, "api_key": raw,
                                "tier": tier, "checkout_url": sess.get("url", "")})

    def _auth(self):
        t = store.authenticate(self._json().get("key", ""))
        if not t:
            return self._send(401, {"error": "invalid key"})
        if not t.active:
            return self._send(402, {"error": "trial_expired",
                                    "message": "Your free trial has ended. Upgrade to keep Aria."})
        return self._send(200, {"tenant_id": t.tenant_id, "tier": t.tier,
                                "role": t.role, "name": t.display_name,
                                "quota_gb": t.quota_bytes // 1024**3,
                                "used_bytes": store.usage_bytes(t),
                                "dedicated": t.dedicated,
                                "trial_days_left": t.trial_days_left,
                                "is_admin": t.role == "admin"})

    def _admin_trials(self):
        """Admin-only: trial tenants (email + expiry) for the n8n trial-digest.
        The registry stores full tenant metadata (email, trial_ends); only the
        raw API key is hashed — so exposing this to an admin is safe."""
        t = self._tenant()
        if not t or t.role != "admin":
            return self._send(403, {"error": "admin only"})
        trials = []
        with store._lock:
            for tt in store._by_hash.values():
                if tt.tier == "trial" and tt.trial_ends:
                    trials.append({"email": tt.email, "tenant_id": tt.tenant_id,
                                   "trial_ends": tt.trial_ends,
                                   "trial_days_left": tt.trial_days_left})
        return self._send(200, {"trials": trials, "count": len(trials)})

    def _chat(self):
        t = self._tenant()
        if not t:
            return self._send(401, {"error": "unauthorized"})
        d = self._json()
        db = store.memory_db(t)
        req_msgs = d.get("messages", [])
        # ensure Aria identity for engines that don't have it trained in
        if ARIA_IDENTITY and not any(m.get("role") == "system" for m in req_msgs):
            req_msgs = [{"role": "system", "content": ARIA_IDENTITY}] + req_msgs
        # layer THIS tenant's private memory onto the prompt (isolated brain)
        msgs = brains().augment(t.tenant_id, db, req_msgs)
        payload = json.dumps({
            "model": "Aria", "messages": msgs,
            "max_tokens": min(int(d.get("max_tokens", 512)), 2048),
        }).encode()
        req = urllib.request.Request(f"{RAW_ENGINE}/chat/completions",
                                    data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "MnemonicAI-Cloud/1.0")
        if RAW_ENGINE_KEY:
            req.add_header("Authorization", f"Bearer {RAW_ENGINE_KEY}")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            # brand: the model's identity is trained-in as "Aerith"; until a
            # proper identity retrain to "Aria", present as Aria everywhere the
            # user sees (model label + any stray self-reference in the text).
            out["model"] = "Aria"
            reply = out.get("choices", [{}])[0].get("message", {}).get("content", "")
            if "Aerith" in reply:
                reply = reply.replace("Aerith", "Aria")
                out["choices"][0]["message"]["content"] = reply
            brains().remember_reply(t.tenant_id, db, reply)  # perceive into their brain
            toks = out.get("usage", {}).get("total_tokens", 0)
            mc.usage(t.tenant_id, "chat", toks)  # dashboard feed (best-effort)
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
            email = ""
            if cust:
                for tt in store._by_hash.values():
                    if tt.tenant_id == tenant_id:
                        tt.stripe_customer = cust
                        email = tt.email
                store._save()
            # feed Mission Control's CRM (billing already applied above)
            status = "canceled" if tier == "free" else "active"
            mc.subscription(tenant_id, email, tier, status, cust or "")
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
