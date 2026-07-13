"""Stripe billing for MnemonicAI Cloud — dependency-free (urllib).

Handles subscription checkout, the customer portal (cancel / update payment),
and webhook signature verification + tier flipping. Keys come from cloud/.env
(gitignored); test mode by default.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.stripe.com/v1"


def load_env(path: str = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), ".env")
    env = {}
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


class Stripe:
    def __init__(self, env: dict = None) -> None:
        self.env = env or load_env()
        mode = self.env.get("STRIPE_MODE", "test")
        self.secret = self.env["STRIPE_TEST_SECRET" if mode == "test"
                               else "STRIPE_LIVE_SECRET"]
        self.pubkey = self.env.get("STRIPE_TEST_PUBKEY" if mode == "test"
                                   else "STRIPE_LIVE_PUBKEY", "")
        self.webhook_secret = self.env.get("STRIPE_WEBHOOK_SECRET", "")
        self.mode = mode

    # ---- raw API ----
    def _call(self, method: str, path: str, params: dict = None) -> dict:
        data = None
        if params:
            data = urllib.parse.urlencode(_flatten(params)).encode()
        req = urllib.request.Request(f"{API}{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.secret}")
        req.add_header("User-Agent", "MnemonicAI-Cloud/1.0")
        if data:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": json.loads(e.read()).get("error", {}), "status": e.code}

    # ---- product / price setup (run once) ----
    def ensure_products(self) -> dict:
        """Create the Starter/Pro products + monthly prices if missing.
        Returns {'starter': price_id, 'pro': price_id}. Idempotent via
        lookup_key on the price."""
        out = {}
        for tier, name, cents in (("starter", "MnemonicAI Starter", 1900),
                                  ("pro", "MnemonicAI Pro", 7900)):
            existing = self._call("GET", f"/prices?lookup_keys[]=mnem_{tier}&limit=1")
            if existing.get("data"):
                out[tier] = existing["data"][0]["id"]
                continue
            prod = self._call("POST", "/products", {"name": name})
            price = self._call("POST", "/prices", {
                "product": prod["id"], "unit_amount": cents, "currency": "usd",
                "recurring[interval]": "month", "lookup_key": f"mnem_{tier}",
                "transfer_lookup_key": "true"})
            out[tier] = price.get("id", "")
        return out

    # ---- checkout ----
    def checkout(self, tier: str, price_id: str, email: str,
                 tenant_id: str, base_url: str) -> dict:
        return self._call("POST", "/checkout/sessions", {
            "mode": "subscription",
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": 1,
            "customer_email": email,
            "client_reference_id": tenant_id,
            "metadata[tenant_id]": tenant_id,
            "metadata[tier]": tier,
            "subscription_data[metadata][tenant_id]": tenant_id,
            "success_url": f"{base_url}/?checkout=success",
            "cancel_url": f"{base_url}/?checkout=cancel"})

    def portal(self, customer_id: str, base_url: str) -> dict:
        """Stripe-hosted portal for cancel + payment-method updates."""
        return self._call("POST", "/billing_portal/sessions", {
            "customer": customer_id, "return_url": f"{base_url}/#account"})

    # ---- webhook verification ----
    def verify_webhook(self, payload: bytes, sig_header: str,
                       tolerance: int = 300) -> dict | None:
        """Verify the Stripe-Signature header; return the event or None."""
        if not self.webhook_secret or not sig_header:
            return None
        parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
        t, v1 = parts.get("t"), parts.get("v1")
        if not t or not v1:
            return None
        if abs(time.time() - int(t)) > tolerance:
            return None  # replay-protection
        signed = f"{t}.".encode() + payload
        expected = hmac.new(self.webhook_secret.encode(), signed,
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            return None
        return json.loads(payload)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Stripe wants bracketed form-encoding; our params are pre-bracketed."""
    return d


# maps webhook event types -> the tier a tenant should end up on
def tier_from_event(event: dict) -> tuple[str, str] | None:
    """Return (tenant_id, new_tier) for subscription lifecycle events."""
    et = event.get("type", "")
    obj = event.get("data", {}).get("object", {})
    meta = obj.get("metadata", {}) or {}
    tenant_id = meta.get("tenant_id") or obj.get("client_reference_id")
    if not tenant_id:
        return None
    if et in ("checkout.session.completed", "customer.subscription.created",
              "customer.subscription.updated"):
        return tenant_id, meta.get("tier", "starter")
    if et in ("customer.subscription.deleted",):
        return tenant_id, "free"
    return None
