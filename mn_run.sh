#!/usr/bin/env bash
# mn_run.sh — single script to run / stop / observe MnemonicAi.
#
# Subcommands:
#   (default)         foreground serve
#   serve             serve in foreground
#   background        detached, writes mn_run.pid
#   stop              stop the background instance
#   status            is it running? on which GPU? what's the port?
#   logs [N]          tail -n N journal lines (default 100)
#   train             run a QLoRA bake (uses GPU TRAIN_GPU)
#   mock              foreground serve with --backend mock (no GPU used)
#   env               print the resolved mn_env.json + GPU role summary
#   adapter-ui        run the adapter rollback web UI
#
set -u
REPO_ROOT="${MN_REPO:-$PWD}"
cd "$REPO_ROOT" || { echo "not a repo"; exit 2; }

VENV="$REPO_ROOT/mnemonicai_venv"
[ -x "$VENV/bin/python" ] || { echo "venv not found at $VENV — run ./mn_install.sh first"; exit 1; }
source "$VENV/bin/activate"

CMD="${1:-serve}"
[ $# -gt 0 ] && shift

# ---------- GPU role assignment ----------
ROLE_FILE="$REPO_ROOT/.mn_gpu_roles"
if [ -f "$ROLE_FILE" ]; then
  source "$ROLE_FILE"
  : "${INFER_GPU:=0}"
  : "${TRAIN_GPU:=0}"
else
  INFER_GPU=0
  TRAIN_GPU=0
  echo "[!] no .mn_gpu_roles — defaulting to GPU 0 for both roles"
fi
export CUDA_VISIBLE_DEVICES_INFER="$INFER_GPU"
export CUDA_VISIBLE_DEVICES_TRAIN="$TRAIN_GPU"

# ---------- paths ----------
PORT="${MN_PORT:-8400}"
HOST="${MN_HOST:-0.0.0.0}"
DATA_DIR="$REPO_ROOT/mnemonicai_data"
LOG_DIR="$DATA_DIR/logs"
PID_FILE="$REPO_ROOT/mn_run.pid"
mkdir -p "$LOG_DIR"

# ---------- helpers ----------
RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[1;33m'; CYN=$'\033[0;36m'; NC=$'\033[0m'
log() { printf "${CYN}[%s]${NC} %s\n" "$(date +%H:%M:%S)" "$*"; }
fail() { log "FATAL: $*"; exit 1; }

ensure_config() {
  [ -f "$REPO_ROOT/config.json" ] || {
    log "no config.json — running install.py"
    "$VENV/bin/python" install.py --model "$REPO_ROOT/models/Aerith" || fail "install.py"
  }
}

# ---------- subcommands ----------
cmd_env() {
  echo "=== mn_env.json ==="
  if [ -f "$REPO_ROOT/mn_env.json" ]; then cat "$REPO_ROOT/mn_env.json"; else echo "(missing)"; fi
  echo
  echo "=== GPU roles ==="
  if [ -f "$ROLE_FILE" ]; then cat "$ROLE_FILE"; else echo "(missing)"; fi
  echo
  echo "=== runtime ==="
  echo "  host:    $(hostname)"
  echo "  port:    $PORT"
  echo "  data:    $DATA_DIR"
  echo "  logs:    $LOG_DIR"
  echo "  pidfile: $PID_FILE"
}

cmd_status() {
  ensure_config
  echo "=== process ==="
  local running=0
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    pid=$(cat "$PID_FILE")
    echo "RUNNING  pid=$pid"
    ps -o pid,etime,rss,cmd -p "$pid" 2>/dev/null | tail -1
    running=1
  elif command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet mnemonicai.service 2>/dev/null; then
    pid=$(systemctl show -p MainPID --value mnemonicai.service 2>/dev/null)
    echo "RUNNING (systemd)  pid=$pid"
    ps -o pid,etime,rss,cmd -p "$pid" 2>/dev/null | tail -1
    running=1
  else
    echo "NOT RUNNING"
    [ -f "$PID_FILE" ] && echo "(stale pidfile $PID_FILE)"
  fi

  echo
  echo "=== GPUs ==="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -3 || echo "  nvidia-smi failed"
  echo
  echo "=== endpoints ==="
  echo "  live monitor:  http://localhost:${PORT}/"
  echo "  OpenAI API:    http://localhost:${PORT}/v1"
  echo "  events (SSE):  http://localhost:${PORT}/events"
  echo "  adapter UI:    http://localhost:8401/admin/adapters"
  echo
  echo "=== recent log lines ==="
  latest=$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -1)
  if [ -n "$latest" ]; then tail -5 "$latest"; else echo "  (no logs yet)"; fi

  # Return non-zero if not running so health watchers trigger
  [ "$running" -eq 1 ]
}

cmd_logs() {
  N="${1:-100}"
  latest=$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | head -1)
  if [ -z "$latest" ]; then echo "no logs in $LOG_DIR yet"; exit 0; fi
  tail -n "$N" "$latest"
}

cmd_foreground() {
  ensure_config
  log "starting MnemonicAi on $HOST:$PORT  (inference on GPU $CUDA_VISIBLE_DEVICES_INFER)"
  exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_INFER" \
    "$VENV/bin/python" start.py \
    --host "$HOST" --port "$PORT" \
    --data-dir "$DATA_DIR" \
    --log-file "$LOG_DIR/mn_$(date +%Y%m%d_%H%M%S).log" \
    "$@"
}

cmd_background() {
  ensure_config
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log "already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  log "starting in background (inference on GPU $CUDA_VISIBLE_DEVICES_INFER)"
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_INFER" \
        "$VENV/bin/python" start.py \
        --host "$HOST" --port "$PORT" \
        --data-dir "$DATA_DIR" \
        --log-file "$LOG_DIR/mn_$(date +%Y%m%d_%H%M%S).log" \
        "$@" \
        > "$LOG_DIR/mn.out" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  cmd_status
  echo
  log "stop with:   ./mn_run.sh stop"
  log "logs with:    ./mn_run.sh logs"
}

cmd_stop() {
  if [ ! -f "$PID_FILE" ]; then echo "not running (no pidfile)"; return 0; fi
  pid=$(cat "$PID_FILE")
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "stale pidfile removed"; return 0; fi
  log "stopping pid $pid (SIGTERM, 10s grace)"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || { log "stopped"; rm -f "$PID_FILE"; return 0; }
    sleep 1
  done
  log "did not stop gracefully, sending SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

cmd_train() {
  ensure_config
  if [ ! -d "$REPO_ROOT/models/Aerith" ]; then
    fail "no Aerith model dir at $REPO_ROOT/models/Aerith"
  fi
  if [ ! -f "$REPO_ROOT/dryrun_train.py" ]; then
    fail "no dryrun_train.py at $REPO_ROOT"
  fi
  log "QLoRA training run. Device = GPU $CUDA_VISIBLE_DEVICES_TRAIN. Adapter -> $DATA_DIR/adapter"
  env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_TRAIN" \
    "$VENV/bin/python" "$REPO_ROOT/dryrun_train.py" 2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
}

cmd_adapter_ui() {
  log "starting adapter rollback UI on 0.0.0.0:8401"
  exec "$VENV/bin/python" "$REPO_ROOT/app/adapter_ui.py"
}

# ---------- dispatch ----------
case "$CMD" in
  serve)         cmd_foreground "$@" ;;
  background)    cmd_background "$@" ;;
  stop)          cmd_stop ;;
  status)        cmd_status ;;
  logs)          cmd_logs "$@" ;;
  train)         cmd_train ;;
  mock)          cmd_foreground --backend mock "$@" ;;
  env)           cmd_env ;;
  adapter-ui)    cmd_adapter_ui ;;
  -h|--help|help) sed -n '2,23p' "$0" ;;
  *)             echo "unknown subcommand: $CMD"; sed -n '2,23p' "$0"; exit 2 ;;
esac
