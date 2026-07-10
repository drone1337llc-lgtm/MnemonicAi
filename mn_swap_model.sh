#!/bin/bash
# =============================================================================
# Swap the model MnemonicAI serves — with ZERO downtime.
#
#   ./mn_swap_model.sh /path/to/model.gguf [/path/to/hf-dir]
#       Swap to a ready-made GGUF. Pass the matching HF safetensors dir as the
#       second arg if you want sleep-training to keep working on the new base.
#
#   ./mn_swap_model.sh /path/to/hf-model-dir
#       Convert an HF model (e.g. a fresh mergekit output) to Q4_K_M GGUF
#       first, then swap. The HF dir is also wired up for sleep-training.
#
# How the swap works (hybrid backend): the standby llama-server slot boots
# with the new model, gets health-checked, traffic flips atomically, and the
# old engine retires after 60s so in-flight requests finish. Clients on :8400
# never see an error. The memory adapter resets (LoRA is tied to the weights
# it was trained on) — sleep-training rebuilds it from the memory DB, which
# is untouched.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_CPP=/home/surge/Documents/llama.cpp
VENV_PY=/home/surge/mnemonicai_venv/bin/python
GGUF_DIR=./models/gguf
API=http://localhost:8400

[ $# -ge 1 ] || { sed -n '3,14p' "$0"; exit 1; }
SRC=$1
MODEL_PATH="${2:-}"

if [ -d "$SRC" ]; then
  # HF safetensors dir → convert + quantize
  NAME=$(basename "$SRC")
  F16="$GGUF_DIR/$NAME-f16.gguf"
  GGUF="$GGUF_DIR/$NAME-Q4_K_M.gguf"
  MODEL_PATH="$SRC"
  if [ -f "$GGUF" ]; then
    echo "=== $GGUF already exists, reusing it."
  else
    mkdir -p "$GGUF_DIR"
    echo "=== Converting $NAME to GGUF (f16 intermediate)..."
    PYTHONPATH="$LLAMA_CPP/gguf-py" "$VENV_PY" \
      "$LLAMA_CPP/convert_hf_to_gguf.py" --outtype f16 --outfile "$F16" "$SRC"
    echo "=== Quantizing to Q4_K_M..."
    "$LLAMA_CPP/build/bin/llama-quantize" "$F16" "$GGUF" Q4_K_M
    rm -f "$F16"
  fi
elif [ -f "$SRC" ]; then
  GGUF="$SRC"
  NAME=$(basename "$SRC" .gguf)
else
  echo "ERROR: '$SRC' is neither a directory nor a file" >&2
  exit 1
fi

echo "=== Requesting zero-downtime swap to $(basename "$GGUF")..."
BODY=$(jq -n --arg g "$(realpath "$GGUF")" \
             --arg m "${MODEL_PATH:+$(realpath "$MODEL_PATH")}" \
             --arg n "$NAME" \
             '{gguf_path:$g, model_name:$n} + (if $m != "" then {model_path:$m} else {} end)')
RESP=$(curl -sf -X POST "$API/admin/swap-base" \
        -H "Content-Type: application/json" -d "$BODY") || {
  echo "ERROR: swap request failed — is MnemonicAI running with the hybrid backend?" >&2
  curl -s "$API/health" >&2 || true
  exit 1
}
echo "$RESP" | jq .
echo "=== Done. Old model keeps serving in-flight requests for 60s, then retires."
