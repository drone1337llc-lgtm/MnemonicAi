"""Blue/green llama-server engine manager for zero-downtime adapter swaps.

Two localhost llama-server slots (ports 8402/8403). Exactly one is "active"
and receives all inference traffic from the hybrid backend. A swap boots the
standby slot with the new LoRA adapter GGUF, health-checks it, atomically
flips the state file, then retires the old engine after a grace period so
in-flight requests finish. Clients (OpenClaw etc.) only ever see MnemonicAi's
stable :8400 API — they never notice the swap.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

PORTS = (8402, 8403)
GRACE_SECONDS = 60


class EngineManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.exe = cfg.llama_server_exe
        self.state_path = os.path.join(cfg.data_dir, "engine_state.json")
        os.makedirs(cfg.data_dir, exist_ok=True)

    # ---- state ----
    def state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return {"active_port": None, "pid": None,
                    "adapter_gguf": None, "adapter_version": 0}

    def _write_state(self, st: dict) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, self.state_path)

    def active_url(self) -> str:
        st = self.state()
        return f"http://127.0.0.1:{st['active_port']}/v1" if st.get("active_port") else ""

    # ---- engine lifecycle ----
    def _spawn(self, port: int, adapter_gguf: str | None) -> subprocess.Popen:
        ctx = getattr(self.cfg, "engine_ctx", 32768)
        par = getattr(self.cfg, "engine_parallel", 2)
        kv = getattr(self.cfg, "engine_kv_type", "q8_0")
        args = [self.exe, "-m", self.cfg.gguf_path,
                "--host", "127.0.0.1", "--port", str(port),
                "-ngl", "999", "-c", str(ctx), "--cache-reuse", "256",
                # unified KV: slots share one context pool instead of splitting
                # it, so 2 concurrent requests batch continuously AND each can
                # still use the full window (Hermes sends 40k+ tokens)
                "--kv-unified",
                "--parallel", str(par),
                # keep reasoning inline in `content` (parity with the
                # transformers backend); MnemonicAI reads content only
                "--reasoning-format", "none",
                "--alias", self.cfg.model_name]
        if kv:
            # quantized KV cache halves context VRAM (needs flash attention)
            args += ["-fa", "on", "-ctk", kv, "-ctv", kv]
        if adapter_gguf:
            args += ["--lora", adapter_gguf]
        env = os.environ.copy()
        # CUDA build needs cudart/cublas; torch ships them
        try:
            import torch
            lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.name == "nt":
                env["PATH"] = lib + os.pathsep + env.get("PATH", "")
            else:
                env["LD_LIBRARY_PATH"] = (
                    lib + os.pathsep + env.get("LD_LIBRARY_PATH", ""))
        except ImportError:
            pass
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            # own process group so the engine survives service reload logic
            # but can still be retired by pid
            kwargs["start_new_session"] = True
        log = open(os.path.join(self.cfg.data_dir, f"engine_{port}.log"), "ab")
        return subprocess.Popen(args, env=env, stdout=log, stderr=log, **kwargs)

    def _healthy(self, port: int, timeout_s: float = 180.0) -> bool:
        deadline = time.time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _alive(self, pid) -> bool:
        if not pid:
            return False
        if os.name == "nt":
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=15).stdout
                return str(pid) in out
            except Exception:
                return False
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False

    def _kill(self, pid) -> None:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, timeout=15)
            except Exception:
                pass
            return
        import signal
        try:
            os.kill(int(pid), signal.SIGTERM)
            for _ in range(20):          # give in-flight work 10s to finish
                time.sleep(0.5)
                os.kill(int(pid), 0)
            os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass  # already gone

    # ---- public ops ----
    def ensure_running(self) -> str:
        """Make sure the active engine is alive; boot one if not."""
        st = self.state()
        if st.get("active_port") and self._alive(st.get("pid")):
            return self.active_url()
        port = st.get("active_port") or PORTS[0]
        proc = self._spawn(port, st.get("adapter_gguf"))
        if not self._healthy(port):
            raise RuntimeError(f"engine on :{port} failed health check "
                               f"(see engine_{port}.log)")
        st.update({"active_port": port, "pid": proc.pid})
        self._write_state(st)
        print(f"[hotswap] engine v{st.get('adapter_version', 0)} "
              f"serving on :{port} (pid {proc.pid})")
        return self.active_url()

    def _free_cuda_cache(self) -> None:
        """Release cached (but unused) CUDA memory held by this process.

        Sleep-training runs transformers in-process and PyTorch keeps freed
        blocks in its allocator cache; that cache counts against the GPU and
        has starved standby-engine boots during swaps."""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def swap_in(self, adapter_gguf: str, version: int) -> None:
        """Blue/green swap: boot standby with adapter, flip, retire old."""
        self._free_cuda_cache()
        st = self.state()
        old_port, old_pid = st.get("active_port"), st.get("pid")
        new_port = PORTS[1] if old_port == PORTS[0] else PORTS[0]
        print(f"[hotswap] booting v{version} on :{new_port} "
              f"(adapter: {os.path.basename(adapter_gguf)})")
        proc = self._spawn(new_port, adapter_gguf)
        if not self._healthy(new_port):
            self._kill(proc.pid)
            raise RuntimeError(
                f"new engine v{version} failed health check on :{new_port}; "
                f"keeping v{st.get('adapter_version', 0)} live")
        self._write_state({"active_port": new_port, "pid": proc.pid,
                           "adapter_gguf": adapter_gguf,
                           "adapter_version": version})
        print(f"[hotswap] flipped to v{version} on :{new_port}; retiring "
              f":{old_port} in {GRACE_SECONDS}s")
        if old_pid:
            import threading
            threading.Timer(GRACE_SECONDS, self._kill, args=(old_pid,)).start()

    def swap_base(self, gguf_path: str) -> None:
        """Blue/green swap to a NEW BASE MODEL with zero downtime.

        The old base's LoRA adapter is not carried over (adapters are tied to
        the weights they were trained on); sleep-training rebuilds memory
        adapters on the new base starting from v1. Raises and keeps the old
        engine live if the new one fails its health check.
        """
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)
        self._free_cuda_cache()
        st = self.state()
        old_port, old_pid = st.get("active_port"), st.get("pid")
        new_port = PORTS[1] if old_port == PORTS[0] else PORTS[0]
        old_gguf = self.cfg.gguf_path
        self.cfg.gguf_path = gguf_path
        print(f"[hotswap] booting new base {os.path.basename(gguf_path)} "
              f"on :{new_port}")
        proc = self._spawn(new_port, adapter_gguf=None)
        if not self._healthy(new_port):
            self._kill(proc.pid)
            self.cfg.gguf_path = old_gguf
            raise RuntimeError(
                f"new base model failed health check on :{new_port}; "
                f"keeping {os.path.basename(old_gguf)} live "
                f"(see engine_{new_port}.log)")
        self._write_state({"active_port": new_port, "pid": proc.pid,
                           "adapter_gguf": None, "adapter_version": 0,
                           "base_gguf": gguf_path})
        print(f"[hotswap] base swapped to {os.path.basename(gguf_path)} on "
              f":{new_port}; retiring :{old_port} in {GRACE_SECONDS}s")
        if old_pid:
            import threading
            threading.Timer(GRACE_SECONDS, self._kill, args=(old_pid,)).start()
