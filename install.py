#!/usr/bin/env python3
"""MnemonicAi — the one file to install.

    python3 install.py                 # install GPU deps, detect CUDA, write config
    python3 install.py --mock          # no heavy deps; run everything in mock mode
    python3 install.py --model /path/to/Aerith   # set model weights path

It is safe to re-run. It never downloads a model — point --model at your
Aerith HF safetensors (or drop them in ./models/Aerith).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

BANNER = "═" * 62


def find_python312() -> str | None:
    """Try to locate a Python 3.12 executable on the system."""
    # Check if 'python3.12' is directly available on PATH
    exe = shutil.which("python3.12") or shutil.which("python")
    if exe:
        try:
            out = subprocess.check_output([exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
            if out == "3.12":
                return exe
        except Exception:
            pass

    # On Windows, try the Python Launcher
    if os.name == 'nt':
        py_launcher = shutil.which("py")
        if py_launcher:
            try:
                # Test if py -3.12 can run successfully
                subprocess.check_output([py_launcher, "-3.12", "-c", "pass"], text=True)
                return py_launcher
            except Exception:
                pass
                
    return None


def ensure_venv():
    """Ensure we are running inside a Python 3.12 virtual environment."""
    venv_dir = os.path.abspath("mnemonicai_venv")
    
    # 1. If we are already running inside the target venv, verify it's the right version
    if sys.executable.startswith(venv_dir):
        if sys.version_info.major == 3 and sys.version_info.minor == 12:
            return  # Perfect, we are inside a Python 3.12 venv.
        else:
            print(f"  ! Current venv is running an incorrect Python version ({sys.version.split()[0]}).")
            print("  Wiping and re-creating environment...")
            shutil.rmtree(venv_dir, ignore_errors=True)

    # 2. We need to create or enter the venv
    print(BANNER)
    print("  Bootstrapping Python 3.12 Virtual Environment ...")
    print(BANNER)

    if not os.path.exists(venv_dir):
        # We need a Python 3.12 host executable to build the venv
        if sys.version_info.major == 3 and sys.version_info.minor == 12:
            base_python = sys.executable
            cmd = [base_python, "-m", "venv", venv_dir]
        else:
            py312_exe = find_python312()
            if not py312_exe:
                print("  ERROR: Python 3.12 was not found on your system.")
                print("  PyTorch and its dependencies are not yet stable on Python 3.14+.")
                print("  Please download and install Python 3.12 from python.org before running.")
                sys.exit(1)
            
            if "py.exe" in py312_exe.lower():
                cmd = [py312_exe, "-3.12", "-m", "venv", venv_dir]
            else:
                cmd = [py312_exe, "-m", "venv", venv_dir]

        print(f"  Creating venv at: {venv_dir}")
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            print(f"  Failed to create virtual environment: {e}")
            sys.exit(1)
    
    # 3. Find the executable inside the newly verified venv
    if os.name == 'nt':
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        
    print(f"  Restarting script inside Python 3.12 venv ...\n")
    os.execv(venv_python, [venv_python] + sys.argv)


def sh(cmd) -> int:
    print("  $ " + " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(f"  (could not run: {e})")
        return 1


def main() -> int:
    # Guarantee we are running inside an isolated Python 3.12 environment
    ensure_venv()

    # Safe to import local modules now
    from mnemonicai.appconfig import AppConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="skip GPU deps; mock backend")
    ap.add_argument("--model", default=None, help="path to Aerith HF weights")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    print(BANNER + "\n  MNEMONICAI installer\n" + BANNER)
    print(f"  Python {sys.version.split()[0]} (in venv: {sys.prefix})")

    cfg = AppConfig.load("config.json")
    if args.model:
        cfg.model_path = args.model
    if args.port:
        cfg.port = args.port

    if args.mock:
        cfg.backend = "mock"
        print("\n  Mock mode selected — no GPU dependencies needed.")
    else:
        print("\n  Setting up PyTorch with CUDA ...")
        # Clear out any default pip setups
        sh([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        
        # Install the specific CUDA 12.4 compatible PyTorch build for Python 3.12
        rc_torch = sh([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])
        
        if rc_torch != 0:
            print("  ! Direct PyTorch wheel installation failed.")
            return 1

        print("\n  Installing remaining GPU dependencies (requirements-gpu.txt) …")
        rc = sh([sys.executable, "-m", "pip", "install", "-r", "requirements-gpu.txt"])
        if rc != 0:
            print("  ! Dependency install did not complete perfectly. Check output above.")
            
        print("\n  Installing fast-path attention libraries …")
        rc_fla = sh([sys.executable, "-m", "pip", "install", "flash-linear-attention"])

        if os.name == "nt":
            # No official Windows wheels for causal-conv1d; the source build
            # needs MSVC + a matching CUDA toolkit and usually fails. MnemonicAi
            # ships a pure-PyTorch fallback (mnemonicai/patch_causal.py) that
            # activates automatically, so we skip the build entirely.
            print("  Skipping causal-conv1d on Windows — using the built-in")
            print("  pure-PyTorch fallback (no compiler needed).")
            rc_causal = 0
            # Triton has no official Windows build, but fla (required by
            # Qwen3.5-hybrid models) hard-imports it. Install the maintained
            # community port, pinned to match torch 2.6 (cu124) above:
            # torch 2.6 ↔ triton 3.2.x. Bump this pin if torch is bumped.
            rc_triton = sh([sys.executable, "-m", "pip", "install",
                            "triton-windows<3.3"])
            if rc_triton != 0:
                print("  ! Warning: triton-windows failed to install.")
                print("    The real model will NOT load without it (fla needs triton);")
                print("    the server will fall back to the mock backend.")
            # llama.cpp CUDA wheel: fast GGUF inference (the transformers
            # Mamba path has no Windows CUDA kernels and is ~3 s/token).
            sh([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                "--extra-index-url",
                "https://abetlen.github.io/llama-cpp-python/whl/cu124",
                "--prefer-binary"])
        else:
            rc_causal = sh([sys.executable, "-m", "pip", "install", "causal-conv1d>=1.4.0"])

        if rc_causal != 0 or rc_fla != 0:
            print("\n  ! Warning: Could not install fast-path libraries.")
            print("    The model will still work via the built-in pure-PyTorch")
            print("    fallback — just slower than the compiled kernels.")
            
        cfg.backend = "auto"
        _check_cuda()

    # locate model weights
    print("\n  Checking model weights …")
    from mnemonicai.backend import resolve_model_dir
    resolved = resolve_model_dir(cfg.model_path)
    if resolved:
        if resolved != cfg.model_path:
            print(f"  ✓ found HF cache layout; resolved to snapshot:\n      {resolved}")
        else:
            print(f"  ✓ found weights at {cfg.model_path}")
        import glob
        n_st = len(glob.glob(os.path.join(resolved, "*.safetensors")))
        print(f"    ({n_st} .safetensors file(s) + config.json present)")
    else:
        print(f"  ! no loadable model under {cfg.model_path}")
        print("    Point --model at the folder that contains config.json + *.safetensors")

    cfg.ensure_dirs()
    cfg.save("config.json")
    print(f"\n  Wrote config.json  (backend={cfg.backend}, port={cfg.port})")
    print(BANNER)
    
    print("  Done! ")
    print('      On Windows: double-click "run.bat"')
    print('   OR  type in a terminal "python3 start.py"\n' + BANNER)
    return 0


def _check_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.device_count()} GPU)")
        else:
            print("  ! torch installed but CUDA not available — will run on CPU/mock.")
    except Exception:
        print("  ! torch not importable yet.")


if __name__ == "__main__":
    raise SystemExit(main())