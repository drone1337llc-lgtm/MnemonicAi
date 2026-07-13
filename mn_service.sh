#!/usr/bin/env bash
# mn_service.sh — install systemd unit + health-watch timer for MnemonicAi.
# Idempotent. Does NOT auto-restart on crash (intentional).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)"
source "$SELF/mn_lib.sh"

# Load GPU roles so we can pass them to systemd
source "$MN_REPO/.mn_gpu_roles" 2>/dev/null || true
: "${INFER_GPU:=0}"

# ---------- preflight ----------
command -v systemctl >/dev/null 2>&1 || { _mn_err "systemctl not found"; exit 1; }
[ -x "$MN_VENV/bin/python" ] || { _mn_err "venv not found at $MN_VENV — run ./mn_install.sh first"; exit 1; }
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
[ -d "$HOME_DIR" ] || { _mn_err "could not determine home dir for $USER_NAME"; exit 1; }

UNIT=/etc/systemd/system/mnemonicai.service
TIMER=/etc/systemd/system/mn-watch.timer
SERVICE=/etc/systemd/system/mn-watch.service

_mn_log "writing $UNIT"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=MnemonicAi — Aerith memory-native LLM
After=network-online.target nvidia-persistenced.service
Wants=network-online.target
Requires=nvidia-persistenced.service

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$MN_REPO
ExecStart=$MN_VENV/bin/python start.py --host $MN_HOST --port $MN_PORT --data-dir $MN_DATA
Restart=no
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=$HOME_DIR/.cache/huggingface
Environment=TRANSFORMERS_OFFLINE=0
Environment=CUDA_VISIBLE_DEVICES=${INFER_GPU}
TimeoutStopSec=30
KillSignal=SIGTERM
FinalKillSignal=SIGKILL
MemoryMax=24G
TasksMax=4096
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

_mn_log "writing $SERVICE"
sudo tee "$SERVICE" >/dev/null <<EOF
[Unit]
Description=Health check for MnemonicAi
After=mnemonicai.service

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$MN_REPO
Environment=PYTHONUNBUFFERED=1
Environment=MN_REPO=$MN_REPO
Environment=MN_ALERT_EMAIL=${MN_ALERT_EMAIL}
Environment=MN_ALERT_WEBHOOK=${MN_ALERT_WEBHOOK}
ExecStart=$SELF/.mn_watch.sh
EOF

_mn_log "writing $TIMER (5-min interval)"
sudo tee "$TIMER" >/dev/null <<EOF
[Unit]
Description=Every-5-minutes health check for MnemonicAi

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=10s
Unit=mn-watch.service

[Install]
WantedBy=timers.target
EOF

# ---------- watch helper ----------
cat > "$SELF/.mn_watch.sh" <<'WATCH'
#!/usr/bin/env bash
set -u
SELF="$(cd "$(dirname "$0")" && pwd)"
source "$SELF/mn_lib.sh"
if ! "$SELF/mn_run.sh" status >/dev/null 2>&1; then
  OUT=$("$SELF/mn_run.sh" status 2>&1 || true)
  mn_alert "mn-watch DOWN" "$OUT"
  echo "ALERT sent: server appears down"
  exit 1
fi
echo "OK at $(date -Iseconds)"
WATCH
chmod +x "$SELF/.mn_watch.sh"

# ---------- enable & start ----------
_mn_log "reloading systemd + enabling units"
sudo systemctl daemon-reload
sudo systemctl enable --now mnemonicai.service
sudo systemctl enable --now mn-watch.timer
sleep 2
_mn_log "current state:"
sudo systemctl --no-pager --full status mnemonicai.service | head -10 || true
echo
sudo systemctl --no-pager --full list-timers mn-watch.timer || true
_mn_ok  "systemd unit + watch timer installed and active"
_mn_log  "operate with:  systemctl {start,stop,restart,status} mnemonicai"
_mn_log  "watch log:     journalctl -u mn-watch -f"
_mn_log  "tail last 200: $SELF/mn_run.sh logs 200"
