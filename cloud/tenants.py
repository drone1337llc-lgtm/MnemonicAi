"""Multi-tenant isolation for MnemonicAI cloud.

Every user is a tenant identified by a secret API key. A tenant owns:
  - an ISOLATED memory database (its own file, never shared),
  - a tier (free | starter | pro) with a storage quota,
  - an upload area for their codebase,
  - a Stripe customer id (billing).

Hard privacy rule enforced here: a request authenticated as tenant A can
ONLY ever touch tenant A's memory DB and files. There is no code path that
lets one tenant read another's data, name, or existence. The brain monitor
stream is filtered to the tenant's own events.

Storage layout (small, persistent — lives on the pod's network volume so it
survives restarts):
    /workspace/tenants/
        registry.json                # key_hash -> tenant record (no raw keys)
        <tenant_id>/
            memory.db                # this user's memories only
            uploads/                 # their codebase (quota-limited)
            adapter/                 # their personal baked LoRA (pro)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

TIERS = {
    "free":    {"quota_gb": 1,  "dedicated_pod": False, "price_id": None},
    "starter": {"quota_gb": 5,  "dedicated_pod": False, "price_id": "price_starter"},
    "pro":     {"quota_gb": 20, "dedicated_pod": True,  "price_id": "price_pro"},
}


def _hash_key(key: str) -> str:
    # store only a hash of the API key, never the raw secret
    return hashlib.sha256(("mnem-tenant:" + key).encode()).hexdigest()


@dataclass
class Tenant:
    tenant_id: str
    key_hash: str
    email: str = ""
    tier: str = "free"
    stripe_customer: str = ""
    created: float = field(default_factory=time.time)
    last_active: float = 0.0
    display_name: str = ""

    @property
    def quota_bytes(self) -> int:
        return TIERS.get(self.tier, TIERS["free"])["quota_gb"] * (1024 ** 3)

    @property
    def dedicated(self) -> bool:
        return TIERS.get(self.tier, TIERS["free"])["dedicated_pod"]


class TenantStore:
    def __init__(self, root: str = "/workspace/tenants") -> None:
        self.root = root
        self.reg_path = os.path.join(root, "registry.json")
        self._lock = threading.RLock()
        self._by_hash: Dict[str, Tenant] = {}
        os.makedirs(root, exist_ok=True)
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not os.path.isfile(self.reg_path):
            return
        try:
            with open(self.reg_path) as f:
                data = json.load(f)
            for rec in data.get("tenants", []):
                t = Tenant(**rec)
                self._by_hash[t.key_hash] = t
        except Exception as e:
            print(f"[tenants] registry load failed: {e}")

    def _save(self) -> None:
        tmp = self.reg_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"tenants": [asdict(t) for t in self._by_hash.values()]},
                      f, indent=2)
        os.replace(tmp, self.reg_path)

    # ---- lifecycle ----
    def create(self, email: str = "", tier: str = "free",
               display_name: str = "") -> tuple[str, Tenant]:
        """Returns (raw_api_key, tenant). The raw key is shown to the user
        ONCE and never stored — only its hash is kept."""
        with self._lock:
            raw = "sk-mnem-" + secrets.token_hex(24)
            tid = "t_" + secrets.token_hex(8)
            t = Tenant(tenant_id=tid, key_hash=_hash_key(raw), email=email,
                       tier=tier, display_name=display_name or email.split("@")[0])
            self._by_hash[t.key_hash] = t
            os.makedirs(self.tenant_dir(t), exist_ok=True)
            os.makedirs(os.path.join(self.tenant_dir(t), "uploads"), exist_ok=True)
            self._save()
            return raw, t

    def authenticate(self, api_key: str) -> Optional[Tenant]:
        """Constant-time lookup; returns the tenant or None. This is the ONLY
        way a request is bound to a tenant — no key, no access."""
        if not api_key:
            return None
        h = _hash_key(api_key.strip())
        with self._lock:
            for kh, t in self._by_hash.items():
                if hmac.compare_digest(kh, h):
                    t.last_active = time.time()
                    return t
        return None

    def set_tier(self, tenant_id: str, tier: str) -> bool:
        with self._lock:
            for t in self._by_hash.values():
                if t.tenant_id == tenant_id:
                    t.tier = tier
                    self._save()
                    return True
        return False

    def by_stripe_customer(self, cust: str) -> Optional[Tenant]:
        with self._lock:
            return next((t for t in self._by_hash.values()
                         if t.stripe_customer == cust), None)

    # ---- per-tenant paths (isolation boundary) ----
    def tenant_dir(self, t: Tenant) -> str:
        return os.path.join(self.root, t.tenant_id)

    def memory_db(self, t: Tenant) -> str:
        return os.path.join(self.tenant_dir(t), "memory.db")

    def uploads_dir(self, t: Tenant) -> str:
        return os.path.join(self.tenant_dir(t), "uploads")

    def adapter_dir(self, t: Tenant) -> str:
        return os.path.join(self.tenant_dir(t), "adapter")

    # ---- quota ----
    def usage_bytes(self, t: Tenant) -> int:
        total = 0
        for dirpath, _, files in os.walk(self.tenant_dir(t)):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total

    def quota_ok(self, t: Tenant, incoming_bytes: int = 0) -> bool:
        return self.usage_bytes(t) + incoming_bytes <= t.quota_bytes
