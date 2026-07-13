#!/usr/bin/env bash
# mn_install.sh — single-script install for the consolidated MnemonicAi box.
# Target: AMD 5950X + RTX 4080 SUPER (GPU 0, inference) + RTX 3090 (GPU 1, training)
# Idempotent. Safe to re-run. Never aborts on a single failure.
#
# Usage:  ./mn_install.sh
#   MN_INFER_GPU=1 MN_TRAIN_GPU=0 ./mn_install.sh   # swap roles
#   MN_SKIP_REBOOT=1 ./mn_install.sh                 # skip auto-reboot
#
set -u

# ---------- pretty output ----------
RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[1;33m'; CYN=$'\033[0;36m'; NC=$'\033[0m'
log()   { printf "${CYN}[%s]${NC} %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()    { printf "${GRN}  ✓${NC} %s\n" "$*"; COMPLETED+=("$1"); }
warn()  { printf "${YLW}  !${NC} %s\n" "$*"; FAILED+=("$1 :: $*"); }
fail()  { printf "${RED}  ✗${NC} %s\n" "$*"; FAILED+=("$1 :: $*"); }
hr()    { printf "${CYN}── %s ──${NC}\n" "$*"; }

COMPLETED=()
FAILED=()

REPO_ROOT="${MN_REPO:-$PWD}"
[ -d "$REPO_ROOT/.git" ] || { echo "Not a git repo at $REPO_ROOT. cd into MnemonicAi/ and re-run."; exit 2; }
cd "$REPO_ROOT" || exit 2
log "Repo: $REPO_ROOT"
log "User: ${SUDO_USER:-$USER}  Host: $(hostname)"

# ---------- step gating ----------
MARKER_DIR="$REPO_ROOT/.mn_install_state"
mkdir -p "$MARKER_DIR"
done_step() { : > "$MARKER_DIR/$1"; }
is_done()   { [ -f "$MARKER_DIR/$1" ]; }

# ==============================================================
# 1/12  OS check
# ==============================================================
hr "1/12  OS check"
. /etc/os-release 2>/dev/null || fail "OS detection" "not an Ubuntu/Debian /etc/os-release"
log "Detected ${PRETTY_NAME:-Linux} (${VERSION_ID:-?})"
case "${VERSION_ID:-}" in
  24.04|22.04) ok "Ubuntu LTS ${VERSION_ID}" ;;
  *)            warn "Ubuntu ${VERSION_ID:-?} — designed for 24.04 / 22.04" ;;
esac

# ==============================================================
# 2/12  Python 3.12
# ==============================================================
hr "2/12  Python 3.12"
if is_done python312; then ok "Python 3.12 (already done)"
else
  if ! command -v python3.12 >/dev/null 2>&1; then
    log "installing python3.12 + venv + dev headers"
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip 2>/dev/null \
      || fail "python3.12 apt" "see /tmp/mn_install.log"
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    ok "Python 3.12 present"
    done_step python312
  else
    fail "python3.12" "not on PATH after install"
  fi
fi
PY=python3.12

# ==============================================================
# 3/12  Git sync
# ==============================================================
hr "3/12  Git sync"
if [ -d "$REPO_ROOT/.git" ]; then
  if git pull --ff-only 2>/dev/null; then ok "git pull --ff-only"
  else warn "git pull" "offline? diverged? — continuing with local checkout"
  fi
else
  warn "git" "no .git/ — continuing with whatever's on disk"
fi

# ==============================================================
# 4/12  venv
# ==============================================================
hr "4/12  Python venv"
VENV="$REPO_ROOT/mnemonicai_venv"
if is_done venv; then ok "venv at $VENV (already done)"
else
  log "creating venv"
  if "$PY" -m venv "$VENV" 2>/dev/null; then
    ok "venv created"
    done_step venv
  else
    fail "venv create" "see $VENV/log"
  fi
fi
source "$VENV/bin/activate"
PY=python
ok "venv activated"

# ==============================================================
# 5/12  pip base + editable
# ==============================================================
hr "5/12  pip base + editable"
if is_done pip-base; then ok "base packages (already done)"
else
  "$PY" -m pip install --upgrade pip wheel setuptools "setuptools<70" 2>/dev/null \
    || warn "pip upgrade" "non-fatal"
  "$PY" -m pip install -r "$REPO_ROOT/requirements.txt" 2>/dev/null \
    && ok "requirements.txt" || fail "requirements.txt" "see pip log"
  "$PY" -m pip install -e "$REPO_ROOT" 2>/dev/null \
    && { ok "MnemonicAi (editable)"; done_step pip-base; } \
    || fail "pip install -e ." "see pip log"
fi

# ==============================================================
# 6/12  GPU stack detection
# ==============================================================
hr "6/12  GPU stack detection"
GPU_STACK="none"
NVIDIA_SMI_OK=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  NVIDIA_SMI_OK=1
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU')
  log "nvidia-smi: yes, ${GPU_COUNT} GPU(s) detected"
  if [ "$GPU_COUNT" -ge 2 ]; then
    log "this box has multiple GPUs — using the .mn_gpu_roles file to assign them"
  else
    log "this box has 1 GPU — both inference and training will use GPU 0"
  fi
fi
if [ "$NVIDIA_SMI_OK" = "1" ]; then
  GPU_STACK="cuda"
  ok "NVIDIA CUDA stack detected"
else
  GPU_STACK="cpu"
  warn "no nvidia-smi" "this box will fall back to CPU/mock — training will be very slow or fail"
fi

# write gpu_stack immediately so Step 12 can read it
if [ -f "$REPO_ROOT/mn_env.json" ]; then
  "$PY" -c "
import json
p = '$REPO_ROOT/mn_env.json'
try:
    with open(p) as f: env = json.load(f)
except Exception:
    env = {}
env['gpu_stack'] = '$GPU_STACK'
with open(p, 'w') as f: json.dump(env, f, indent=2, default=str)
" 2>/dev/null || true
fi

# ==============================================================
# 7/12  NVIDIA driver + CUDA toolkit
# ==============================================================
hr "7/12  NVIDIA driver + CUDA toolkit"
if is_done nvidia-driver; then ok "NVIDIA driver (already done)"
else
  log "installing nvidia-driver-560 + cuda-toolkit-12-6 (apt)"
  sudo apt-get install -y -qq software-properties-common gnupg 2>/dev/null \
    || warn "apt-add-repo prep" "non-fatal"
  sudo add-apt-repository -y ppa:graphics-drivers/ppa 2>/dev/null \
    || warn "graphics-drivers PPA" "may already exist"
  sudo apt-get update -qq
  if sudo apt-get install -y -qq nvidia-driver-560 cuda-toolkit-12-6 2>/dev/null; then
    ok "nvidia-driver-560 + cuda-toolkit-12-6"
    done_step nvidia-driver
    if [ "${MN_SKIP_REBOOT:-0}" != "1" ]; then
      echo
      log "REBOOT required for the new NVIDIA driver. After reboot, re-run:"
      log "  cd $REPO_ROOT && ./mn_install.sh"
      sudo reboot
    fi
  else
    fail "NVIDIA driver install" "may need a reboot; see /tmp/mn_install.log"
  fi
fi
nvidia-smi | head -8 || warn "nvidia-smi" "driver may not be loaded"
nvcc --version 2>/dev/null | tail -2 || warn "nvcc" "CUDA toolkit not on PATH"

# ==============================================================
# 8/12  PyTorch + GPU stack (CUDA)
# ==============================================================
hr "8/12  PyTorch + GPU stack (CUDA)"
if is_done torch-stack; then ok "torch stack (already done)"
else
  "$PY" -m pip uninstall -y -q torch torchvision torchaudio bitsandbytes \
      accelerate peft transformers safetensors 2>/dev/null || true

  log "installing PyTorch 2.6.0 + cu126"
  if "$PY" -m pip install --upgrade \
      --index-url https://download.pytorch.org/whl/cu126 \
      "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" 2>/dev/null; then
    ok "PyTorch 2.6.0+cu126"
  else
    fail "PyTorch cu126" "see pip log"
  fi

  log "installing transformers/PEFT/bnb from requirements-gpu.txt"
  if "$PY" -m pip install --upgrade -r "$REPO_ROOT/requirements-gpu.txt" 2>/dev/null; then
    ok "transformers/peft/bnb/accelerate/safetensors/sentencepiece"
  else
    fail "requirements-gpu.txt" "see pip log"
  fi

  log "building llama-cpp-python with CUDA (this takes ~10-15 minutes)"
  if CMAKE_ARGS="-DGGML_CUDA=on" "$PY" -m pip install --upgrade --force-reinstall --no-cache-dir \
      "llama-cpp-python[cuda]" 2>/dev/null; then
    ok "llama-cpp-python (CUDA)"
  else
    warn "llama-cpp-python (CUDA)" "inference will fall back to HF transformers"
  fi
  done_step torch-stack
fi

# ==============================================================
# 9/12  Multi-GPU smoke test
# ==============================================================
hr "9/12  Multi-GPU smoke test"
"$PY" - <<'PY' 2>&1 | tee /tmp/mn_smoke.log
import sys, traceback
try:
    import torch
    print(f"torch={torch.__version__}  cuda-runtime={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("CUDA not available — this is a CPU-only box")
        sys.exit(0)
    n = torch.cuda.device_count()
    print(f"GPUs visible: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap  = torch.cuda.get_device_capability(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i}  {name}  (sm_{cap[0]}{cap[1]}, {free/1e9:.1f}/{total/1e9:.1f} GB free)")
    with torch.cuda.device(0):
        m = torch.nn.Linear(8, 4).cuda()
        x = torch.randn(2, 8, device="cuda", requires_grad=True)
        y = m(x).sum(); y.backward(); torch.cuda.synchronize()
        print("CUDA forward+backward on cuda:0: OK")
    if n >= 2:
        with torch.cuda.device(1):
            m = torch.nn.Linear(8, 4).cuda()
            x = torch.randn(2, 8, device="cuda", requires_grad=True)
            y = m(x).sum(); y.backward(); torch.cuda.synchronize()
            print("CUDA forward+backward on cuda:1: OK")
    print("multi-GPU smoke test: OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY
[ ${PIPESTATUS[0]} -eq 0 ] && ok "multi-GPU smoke test" || warn "multi-GPU smoke test" "non-fatal"

# ==============================================================
# 10/12  PCIe topology check
# ==============================================================
hr "10/12  PCIe topology check"
"$PY" - <<'PY' 2>&1 | tee -a /tmp/mn_smoke.log
import subprocess
out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,pcie.link.width.gen,pcie.link.width.current",
                       "--format=csv,noheader,nounits"], capture_output=True, text=True)
print(out.stdout)
bad = [l for l in out.stdout.splitlines() if "x4" in l or " 4 " in l]
if bad:
    print("!! one or more GPUs are running at PCIe x4 — this will throttle training.")
    print("   recommended: move the affected card to a CPU-fed x8/x16 slot.")
PY
ok "PCIe topology inspected"

# ==============================================================
# 11/12  Model weights + install.py + train_check.py
# ==============================================================
hr "11/12  Model weights + project wiring"
MODEL_DIR="$REPO_ROOT/models/Aerith"
GGUF_DIR="$REPO_ROOT/models/gguf"
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
  ok "raw HF weights at $MODEL_DIR"
else
  fail "model weights" "missing $MODEL_DIR — place the Aerith safetensors here (current: /home/surge/Documents/mergekit/models/Aerith)"
fi
if ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || ls "$MODEL_DIR"/*.bin >/dev/null 2>&1; then
  ok "model weights files found in $MODEL_DIR"
else
  warn "model weights" "no .safetensors/.bin found in $MODEL_DIR"
fi
if ls "$GGUF_DIR"/*.gguf >/dev/null 2>&1; then
  ok "GGUF at $GGUF_DIR ($(ls "$GGUF_DIR"/*.gguf | wc -l) file(s))"
else
  warn "GGUF" "no .gguf in $GGUF_DIR — install.py will leave cfg.gguf_path as-is; HF inference will be used"
fi
if [ -f "$REPO_ROOT/install.py" ]; then
  if "$PY" install.py --model "$MODEL_DIR" > /tmp/mn_install_py.log 2>&1; then
    tail -8 /tmp/mn_install_py.log
    ok "install.py wrote config.json"
  else
    tail -8 /tmp/mn_install_py.log
    fail "install.py" "see /tmp/mn_install_py.log"
  fi
else
  fail "install.py" "missing — wrong repo?"
fi
if [ -f "$REPO_ROOT/train_check.py" ]; then
  if "$PY" "$REPO_ROOT/train_check.py" > /tmp/mn_train_check.log 2>&1; then
    tail -3 /tmp/mn_train_check.log
    if grep -q "TRAIN_STACK_OK\|OK" /tmp/mn_train_check.log; then
      ok "train_check.py passes"
    else
      warn "train_check" "no OK in tail — see /tmp/mn_train_check.log"
    fi
  else
    tail -3 /tmp/mn_train_check.log
    warn "train_check" "non-zero exit — see /tmp/mn_train_check.log"
  fi
else
  warn "train_check.py" "missing (OK if you don't need to bake on this box)"
fi

# ==============================================================
# 12/12  Enumerate GPUs + write mn_env.json + GPU role file
# ==============================================================
hr "12/12  Enumerate GPUs + mn_env.json + GPU roles"
"$PY" - <<PY 2>&1 | tee -a /tmp/mn_install.log
import json, subprocess, os
p = "$REPO_ROOT/mn_env.json"
try:
    with open(p) as f: env = json.load(f)
except Exception:
    env = {}
env["gpu_stack"] = env.get("gpu_stack", "$GPU_STACK")
env["gpus"] = []
try:
    if env.get("gpu_stack") == "cuda":
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.total,memory.free,"
             "pcie.link.width.gen,pcie.link.width.current,driver_version",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            idx, name, util, mtot, mfree, lw, lwcur, drv = [x.strip() for x in line.split(",")]
            env["gpus"].append({
                "index": int(idx),
                "name": name,
                "mem_total_mb": int(float(mtot)) if mtot else None,
                "pcie_link": f"{lwcur} lanes @ PCIe Gen{lw}" if lw else None,
                "driver": drv,
            })
    elif env.get("gpu_stack") == "intel-xpu":
        for d in sorted(os.listdir("/sys/class/drm")):
            if not d.startswith("card") or "-" in d: continue
            ven = open(f"/sys/class/drm/{d}/device/vendor").read().strip()
            if ven == "0x8086":
                env["gpus"].append({"index": len(env["gpus"]), "name": "Intel Arc"})
    print(f"detected {len(env['gpus'])} GPU(s)")
    for g in env["gpus"]:
        link = g.get('pcie_link') or 'unknown'
        print(f"  cuda:{g['index']}  {g['name']}  ({link})")
except Exception as e:
    print(f"gpu enumeration failed: {e}")
with open(p, "w") as f: json.dump(env, f, indent=2, default=str)
PY

# GPU role file
INFER_GPU="${MN_INFER_GPU:-0}"
TRAIN_GPU="${MN_TRAIN_GPU:-1}"
N_GPUS=$(grep -c '"index"' "$REPO_ROOT/mn_env.json" 2>/dev/null || echo 0)
if [ "$N_GPUS" -lt 2 ] && [ "$TRAIN_GPU" != "$INFER_GPU" ]; then
  warn "GPU roles" "only $N_GPUS GPU(s) visible — both roles will use GPU $INFER_GPU"
  TRAIN_GPU="$INFER_GPU"
fi
cat > "$REPO_ROOT/.mn_gpu_roles" <<EOF
# auto-generated by mn_install.sh — edit and re-source to swap roles
INFER_GPU=${INFER_GPU}
TRAIN_GPU=${TRAIN_GPU}
EOF
cat "$REPO_ROOT/.mn_gpu_roles"
ok "GPU roles assigned"

# ==============================================================
# SUMMARY
# ==============================================================
hr "SUMMARY"
printf "  ${GRN}%-6s${NC} %d\n" "OK"     "${#COMPLETED[@]}"
printf "  ${RED}%-6s${NC} %d\n" "FAIL"   "${#FAILED[@]}"
echo
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "  Failed steps:"
  printf "    - %s\n" "${FAILED[@]}"
  echo
  echo "  Recovery:"
  echo "    cat /tmp/mn_install.log | grep -E 'FAIL|WARN' | less"
  echo "    fix the items above (most are non-fatal — see warnings)"
  echo "    ./mn_install.sh     # idempotent — picks up where it left off"
fi
echo
echo "  GPU stack:   $GPU_STACK"
echo "  GPUs:        ${N_GPUS:-0}"
echo "  Inference:   GPU ${INFER_GPU}"
echo "  Training:    GPU ${TRAIN_GPU}"
echo
echo "  Next:        ./mn_run.sh            # foreground serve"
echo "                ./mn_run.sh background # detached, survives ssh close"
echo "                ./mn_run.sh train      # QLoRA bake (uses GPU ${TRAIN_GPU})"
