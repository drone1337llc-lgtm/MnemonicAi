"""Mission Control usage emitter.

The gateway is the billing source of truth; Mission Control's dashboard wants
per-client subscription + usage data without its own Stripe integration. This
pushes events to mission-api's business endpoint — fire-and-forget, best
effort, never blocks or breaks a chat/webhook (analytics must never gate
billing or serving).

Config (cloud/.env):
    MC_USAGE_URL   = http://192.168.68.64:20501/business/usage
    MC_BUSINESS_KEY = <MC_BUSINESS_KEY from TrueNAS>
If either is unset, this is a silent no-op.

Contract (documented in Desktop/AERITH-CLOUD-STATUS.md for the MC Claude):
  POST {MC_USAGE_URL}  header X-API-Key: {MC_BUSINESS_KEY}
  subscription event: {"type":"subscription","tenant_id","email","tier",
                       "status","stripe_customer","ts"}
  usage event:        {"type":"usage","tenant_id","event","tokens?","ts"}
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request


class MCEmitter:
    def __init__(self, url: str = "", key: str = "") -> None:
        self.url = url
        self.key = key

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _post(self, body: dict) -> None:
        try:
            data = json.dumps(body).encode()
            req = urllib.request.Request(self.url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("X-API-Key", self.key)
            req.add_header("User-Agent", "MnemonicAI-Cloud/1.0")
            urllib.request.urlopen(req, timeout=8).read()
        except Exception:
            pass  # analytics is best-effort; never surface to the caller

    def _fire(self, body: dict) -> None:
        if not self.enabled:
            return
        body.setdefault("ts", int(time.time()))
        threading.Thread(target=self._post, args=(body,), daemon=True).start()

    # ---- public ----
    def subscription(self, tenant_id: str, email: str, tier: str,
                     status: str, stripe_customer: str = "") -> None:
        self._fire({"type": "subscription", "tenant_id": tenant_id,
                    "email": email, "tier": tier, "status": status,
                    "stripe_customer": stripe_customer})

    def usage(self, tenant_id: str, event: str, tokens: int = 0) -> None:
        b = {"type": "usage", "tenant_id": tenant_id, "event": event}
        if tokens:
            b["tokens"] = tokens
        self._fire(b)
