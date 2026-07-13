"""Pro-tier dedicated pod lifecycle.

A Pro tenant gets their own GPU pod: created on first sign-in, stopped after
30 minutes of inactivity to control cost. The controller records each Pro
tenant's pod + last-active time and reconciles on a timer.

RunPod REST API. The API key lives in the env (never committed). Pods are
tagged with the tenant id so we never touch the wrong one.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

RUNPOD_API = "https://rest.runpod.io/v1"
IDLE_STOP_S = 30 * 60


class ProPods:
    def __init__(self, runpod_key: str, image: str, gpu: str = "NVIDIA A40",
                 poll_s: int = 120) -> None:
        self.key = runpod_key
        self.image = image
        self.gpu = gpu
        self._pods: dict[str, dict] = {}     # tenant_id -> {pod_id, url, last}
        self._lock = threading.RLock()
        threading.Thread(target=self._reap_loop, args=(poll_s,),
                         daemon=True).start()

    def _api(self, method: str, path: str, body: dict = None) -> dict:
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(f"{RUNPOD_API}{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read() or b"{}")
        except Exception as e:
            return {"error": str(e)[:120]}

    def ensure(self, tenant_id: str) -> dict:
        """Return the tenant's live pod, creating it if needed. Bumps activity."""
        with self._lock:
            rec = self._pods.get(tenant_id)
            if rec and rec.get("pod_id"):
                rec["last"] = time.time()
                return rec
            pod = self._api("POST", "/pods", {
                "name": f"aerith-pro-{tenant_id}",
                "imageName": self.image,
                "gpuTypeIds": [self.gpu], "gpuCount": 1,
                "containerDiskInGb": 30, "volumeInGb": 60,
                "volumeMountPath": "/workspace",
                "ports": ["8400/http"],
                "env": {"TENANT_ID": tenant_id}})
            pid = pod.get("id", "")
            rec = {"pod_id": pid, "last": time.time(),
                   "url": f"https://{pid}-8400.proxy.runpod.net" if pid else ""}
            self._pods[tenant_id] = rec
            return rec

    def touch(self, tenant_id: str) -> None:
        with self._lock:
            if tenant_id in self._pods:
                self._pods[tenant_id]["last"] = time.time()

    def stop(self, tenant_id: str) -> None:
        with self._lock:
            rec = self._pods.pop(tenant_id, None)
        if rec and rec.get("pod_id"):
            self._api("DELETE", f"/pods/{rec['pod_id']}")

    def _reap_loop(self, poll_s: int) -> None:
        while True:
            time.sleep(poll_s)
            now = time.time()
            with self._lock:
                idle = [tid for tid, r in self._pods.items()
                        if now - r["last"] > IDLE_STOP_S]
            for tid in idle:
                print(f"[propods] tenant {tid} idle >30m — stopping pod")
                self.stop(tid)
