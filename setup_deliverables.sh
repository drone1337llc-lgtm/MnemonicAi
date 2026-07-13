#!/usr/bin/env bash
# setup_deliverables.sh — creates all K8s manifests, CI/CD workflows,
# Prometheus/Grafana configs, and observability compose file.
# Run from the MnemonicAi repo root:  bash setup_deliverables.sh
set -euo pipefail

echo "=== Creating deliverable files ==="

# ============================================================
# k8s/namespace.yaml
# ============================================================
mkdir -p k8s
cat > k8s/namespace.yaml <<'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: mnemonicai
  labels:
    app.kubernetes.io/name: mnemonicai
    app.kubernetes.io/part-of: mnemonicai
EOF

# ============================================================
# k8s/configmap.yaml
# ============================================================
cat > k8s/configmap.yaml <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: mnemonicai-config
  namespace: mnemonicai
data:
  config.json: |
    {
      "model_path": "models/Aerith",
      "gguf_path": "models/gguf/Aerith-Q4_K_M.gguf",
      "adapter_dir": "mnemonicai_data/adapter",
      "data_dir": "mnemonicai_data",
      "train_lr": 2e-4,
      "train_steps": 8,
      "lora_r": 16,
      "lora_alpha": 64,
      "lora_dropout": 0.05,
      "lora_target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
      "qlora_batch_size": 4,
      "qlora_grad_accum": 4,
      "qlora_gradient_checkpointing": true,
      "qlora_max_seq_length": 1024,
      "qlora_packing": true,
      "qlora_max_grad_norm": 0.3,
      "qlora_warmup_ratio": 0.03,
      "qlora_lr_scheduler": "cosine",
      "qlora_weight_decay": 0.0,
      "qlora_dataloader_workers": 2,
      "qlora_seed": 42,
      "qlora_optim": "paged_adamw_8bit"
    }
EOF

# ============================================================
# k8s/pvc.yaml
# ============================================================
cat > k8s/pvc.yaml <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mnemonicai-data
  namespace: mnemonicai
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mnemonicai-models
  namespace: mnemonicai
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
EOF

# ============================================================
# k8s/deployment.yaml
# ============================================================
cat > k8s/deployment.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnemonicai
  namespace: mnemonicai
  labels:
    app: mnemonicai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mnemonicai
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: mnemonicai
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: mnemonicai
          image: ghcr.io/YOUR_GITHUB_ORG/mnemonicai:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8400
              name: api
            - containerPort: 8401
              name: adapter-ui
          env:
            - name: MN_REPO
              value: "/app"
            - name: MN_HOST
              value: "0.0.0.0"
            - name: MN_PORT
              value: "8400"
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: HF_HOME
              value: "/app/.cache/huggingface"
            - name: TRANSFORMERS_OFFLINE
              value: "0"
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: 16Gi
              cpu: "8"
            limits:
              nvidia.com/gpu: 1
              memory: 32Gi
              cpu: "16"
          volumeMounts:
            - name: data
              mountPath: /app/mnemonicai_data
            - name: models
              mountPath: /app/models
            - name: config
              mountPath: /app/config.json
              subPath: config.json
              readOnly: true
            - name: dshm
              mountPath: /dev/shm
          readinessProbe:
            httpGet:
              path: /health
              port: 8400
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8400
            initialDelaySeconds: 60
            periodSeconds: 30
            failureThreshold: 3
          lifecycle:
            preStop:
              exec:
                command: ["./mn_run.sh", "stop"]
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: mnemonicai-data
        - name: models
          persistentVolumeClaim:
            claimName: mnemonicai-models
        - name: config
          configMap:
            name: mnemonicai-config
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
EOF

# ============================================================
# k8s/service.yaml
# ============================================================
cat > k8s/service.yaml <<'EOF'
apiVersion: v1
kind: Service
metadata:
  name: mnemonicai
  namespace: mnemonicai
  labels:
    app: mnemonicai
spec:
  type: ClusterIP
  selector:
    app: mnemonicai
  ports:
    - name: api
      port: 8400
      targetPort: 8400
      protocol: TCP
    - name: adapter-ui
      port: 8401
      targetPort: 8401
      protocol: TCP
---
apiVersion: v1
kind: Service
metadata:
  name: mnemonicai-external
  namespace: mnemonicai
  labels:
    app: mnemonicai
spec:
  type: LoadBalancer
  selector:
    app: mnemonicai
  ports:
    - name: api
      port: 80
      targetPort: 8400
      protocol: TCP
EOF

# ============================================================
# k8s/servicemonitor.yaml
# ============================================================
cat > k8s/servicemonitor.yaml <<'EOF'
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mnemonicai
  namespace: mnemonicai
  labels:
    app: mnemonicai
    release: prometheus
spec:
  selector:
    matchLabels:
      app: mnemonicai
  namespaceSelector:
    matchNames:
      - mnemonicai
  endpoints:
    - port: api
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
EOF

# ============================================================
# k8s/train-job.yaml
# ============================================================
cat > k8s/train-job.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: mnemonicai-train-run
  namespace: mnemonicai
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: OnFailure
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: trainer
          image: ghcr.io/YOUR_GITHUB_ORG/mnemonicai:latest
          imagePullPolicy: Always
          command: ["./mn_run.sh", "train"]
          env:
            - name: MN_REPO
              value: "/app"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: 16Gi
            limits:
              nvidia.com/gpu: 1
              memory: 32Gi
          volumeMounts:
            - name: data
              mountPath: /app/mnemonicai_data
            - name: models
              mountPath: /app/models
            - name: config
              mountPath: /app/config.json
              subPath: config.json
              readOnly: true
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: mnemonicai-data
        - name: models
          persistentVolumeClaim:
            claimName: mnemonicai-models
        - name: config
          configMap:
            name: mnemonicai-config
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
EOF

# ============================================================
# k8s/kustomization.yaml
# ============================================================
cat > k8s/kustomization.yaml <<'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: mnemonicai
resources:
  - namespace.yaml
  - configmap.yaml
  - pvc.yaml
  - deployment.yaml
  - service.yaml
  - servicemonitor.yaml
images:
  - name: ghcr.io/YOUR_GITHUB_ORG/mnemonicai
    newTag: latest
EOF

# ============================================================
# .github/workflows/docker-build-push.yaml
# ============================================================
mkdir -p .github/workflows
cat > .github/workflows/docker-build-push.yaml <<'EOF'
name: Build & Push Docker Image

on:
  push:
    branches:
      - main
    tags:
      - "v*.*.*"
  workflow_dispatch:
    inputs:
      tag:
        description: "Image tag override (default: commit SHA)"
        required: false
        default: ""

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

permissions:
  contents: read
  packages: write

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=tag
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

  smoke-test:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull and smoke-test image
        run: |
          IMAGE="${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest"
          docker pull "$IMAGE"
          timeout 30 docker run --rm -e MN_REPO=/app -p 8400:8400 "$IMAGE" \
            python start.py --host 0.0.0.0 --port 8400 --backend mock &
          sleep 10
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8400/health || echo "000")
          echo "Health check: $STATUS"
          [ "$STATUS" = "200" ] || { echo "FAIL"; docker logs $(docker ps -q --latest) || true; exit 1; }
          echo "OK: smoke test passed"

      - name: Clean up
        if: always()
        run: docker stop $(docker ps -q) 2>/dev/null || true
EOF

# ============================================================
# .github/workflows/release.yaml
# ============================================================
cat > .github/workflows/release.yaml <<'EOF'
name: Release

on:
  push:
    tags:
      - "v*.*.*"

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate release notes
        id: notes
        run: |
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
          if [ -n "$PREV_TAG" ]; then
            NOTES=$(git log ${PREV_TAG}..HEAD --pretty=format:"- %s" --no-merges)
          else
            NOTES=$(git log --pretty=format:"- %s" --no-merges | head -20)
          fi
          echo "notes<<EOF" >> $GITHUB_OUTPUT
          echo "$NOTES" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          body: |
            ## MnemonicAi ${{ github.ref_name }}

            ### Changes
            ${{ steps.notes.outputs.notes }}

            ### Docker image
            ```
            docker pull ghcr.io/${{ github.repository }}:${{ github.ref_name }}
            ```
          draft: false
          prerelease: ${{ contains(github.ref_name, '-') }}
EOF

# ============================================================
# app/metrics.py
# ============================================================
mkdir -p app
cat > app/metrics.py <<'EOF'
"""
app/metrics.py — Prometheus instrumentation for MnemonicAi.
Import from start.py:  from app.metrics import setup_metrics, update_adapter_metrics
"""
import os
import json
from pathlib import Path
from prometheus_client import (
    Counter, Histogram, Gauge, CONTENT_TYPE_LATEST
)
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI

INFERENCE_REQUESTS = Counter(
    "mnemonicai_inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"]
)

INFERENCE_LATENCY = Histogram(
    "mnemonicai_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

TRAINING_EVENTS = Counter(
    "mnemonicai_training_events_total",
    "Total training (QLoRA bake) events",
    ["result"]
)

TRAINING_LOSS = Gauge("mnemonicai_training_loss", "Last training loss")
EVAL_LOSS = Gauge("mnemonicai_eval_loss", "Last evaluation loss")
ACTIVE_ADAPTER = Gauge("mnemonicai_active_adapter_version", "Active adapter version number")
ADAPTER_COUNT = Gauge("mnemonicai_adapter_count", "Total registered adapter versions")
GPU_MEMORY_USED = Gauge("mnemonicai_gpu_memory_used_bytes", "GPU memory used", ["gpu_index"])
GPU_MEMORY_TOTAL = Gauge("mnemonicai_gpu_memory_total_bytes", "GPU total memory", ["gpu_index"])
GPU_UTILIZATION = Gauge("mnemonicai_gpu_utilization_percent", "GPU utilization %", ["gpu_index"])
HEALTH_STATUS = Gauge("mnemonicai_health_status", "1=healthy, 0=unhealthy")


def update_adapter_metrics(adapter_dir: str):
    registry_path = Path(adapter_dir) / ".registry.json"
    if not registry_path.exists():
        ADAPTER_COUNT.set(0)
        return
    try:
        data = json.loads(registry_path.read_text())
        ADAPTER_COUNT.set(len(data.get("versions", [])))
        active = data.get("active")
        if active and active.startswith("v"):
            ACTIVE_ADAPTER.set(int(active[1:].split(".")[0]))
    except Exception:
        pass


def setup_metrics(app: FastAPI):
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/admin/adapters"],
        inprogress_name="mnemonicai_inprogress",
        inprogress_labels=True,
    ).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False, should_gzip=True
    )

    @app.get("/health", include_in_schema=False)
    async def health():
        HEALTH_STATUS.set(1)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    used, total = torch.cuda.mem_get_info(i)
                    GPU_MEMORY_USED.labels(gpu_index=str(i)).set(used)
                    GPU_MEMORY_TOTAL.labels(gpu_index=str(i)).set(total)
        except Exception:
            pass
        return {"status": "healthy"}
EOF

# ============================================================
# prometheus/prometheus.yml
# ============================================================
mkdir -p prometheus
cat > prometheus/prometheus.yml <<'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: "mnemonicai"
    metrics_path: /metrics
    static_configs:
      - targets: ["mnemonicai:8400"]
        labels:
          service: mnemonicai
    scrape_interval: 10s
    scrape_timeout: 5s

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
EOF

# ============================================================
# prometheus/alerts.yml
# ============================================================
cat > prometheus/alerts.yml <<'EOF'
groups:
  - name: mnemonicai_alerts
    rules:
      - alert: MnemonicAiDown
        expr: up{job="mnemonicai"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MnemonicAi is down"
          description: "MnemonicAi has been unreachable for 1 minute."

      - alert: MnemonicAiHighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="mnemonicai"}[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MnemonicAi P95 latency > 5s"
          description: "95th percentile latency is {{ $value }}s"

      - alert: MnemonicAiHighErrorRate
        expr: |
          100 * sum(rate(http_requests_total{job="mnemonicai",status="5xx"}[5m]))
          / sum(rate(http_requests_total{job="mnemonicai"}[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "MnemonicAi error rate > 10%"
          description: "Error rate is {{ $value }}%"

      - alert: MnemonicAiTrainingFailure
        expr: increase(mnemonicai_training_events_total{result="failure"}[1h]) > 0
        for: 0s
        labels:
          severity: warning
        annotations:
          summary: "Training failure detected"
          description: "A QLoRA training run failed in the last hour."

      - alert: MnemonicAiGPUMemoryHigh
        expr: mnemonicai_gpu_memory_used_bytes / mnemonicai_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory > 95% used"
          description: "GPU {{ $labels.gpu_index }} memory near capacity"

      - alert: MnemonicAiAdapterRollback
        expr: increase(mnemonicai_training_events_total{result="rollback"}[1h]) > 0
        for: 0s
        labels:
          severity: info
        annotations:
          summary: "Adapter rollback occurred"
          description: "An adapter rollback was triggered in the last hour."
EOF

# ============================================================
# grafana/dashboard.json
# ============================================================
mkdir -p grafana
cat > grafana/dashboard.json <<'EOF'
{
  "dashboard": {
    "id": null,
    "uid": "mnemonicai",
    "title": "MnemonicAi — Inference & Training",
    "tags": ["mnemonicai", "llm"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": { "from": "now-1h", "to": "now" },
    "panels": [
      {
        "id": 1, "title": "Request Rate (req/s)", "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 0, "y": 0 },
        "targets": [{ "expr": "sum(rate(http_requests_total{job=\"mnemonicai\"}[5m]))", "legendFormat": "req/s" }],
        "fieldConfig": { "defaults": { "unit": "reqps" } }
      },
      {
        "id": 2, "title": "Error Rate (%)", "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 6, "y": 0 },
        "targets": [{ "expr": "100 * sum(rate(http_requests_total{job=\"mnemonicai\",status=\"5xx\"}[5m])) / sum(rate(http_requests_total{job=\"mnemonicai\"}[5m]))", "legendFormat": "errors %" }],
        "fieldConfig": { "defaults": { "unit": "percent", "thresholds": { "mode": "absolute", "steps": [{"color":"green","value":null},{"color":"yellow","value":5},{"color":"red","value":10}] } } }
      },
      {
        "id": 3, "title": "P95 Latency (s)", "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 12, "y": 0 },
        "targets": [{ "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=\"mnemonicai\"}[5m])) by (le))", "legendFormat": "p95" }],
        "fieldConfig": { "defaults": { "unit": "s", "decimals": 2 } }
      },
      {
        "id": 4, "title": "Active Adapter Version", "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 18, "y": 0 },
        "targets": [{ "expr": "mnemonicai_active_adapter_version", "legendFormat": "v" }],
        "fieldConfig": { "defaults": { "decimals": 0 } }
      },
      {
        "id": 10, "title": "Request Rate Over Time", "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 4 },
        "targets": [{ "expr": "sum by (handler) (rate(http_requests_total{job=\"mnemonicai\"}[5m]))", "legendFormat": "{{handler}}" }],
        "fieldConfig": { "defaults": { "unit": "reqps" } }
      },
      {
        "id": 11, "title": "Latency Distribution (P50/P95/P99)", "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 4 },
        "targets": [
          { "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job=\"mnemonicai\"}[5m])) by (le))", "legendFormat": "p50" },
          { "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=\"mnemonicai\"}[5m])) by (le))", "legendFormat": "p95" },
          { "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=\"mnemonicai\"}[5m])) by (le))", "legendFormat": "p99" }
        ],
        "fieldConfig": { "defaults": { "unit": "s", "decimals": 3 } }
      },
      {
        "id": 20, "title": "Training Loss", "type": "timeseries",
        "gridPos": { "h": 8, "w": 8, "x": 0, "y": 12 },
        "targets": [
          { "expr": "mnemonicai_training_loss", "legendFormat": "train loss" },
          { "expr": "mnemonicai_eval_loss", "legendFormat": "eval loss" }
        ],
        "fieldConfig": { "defaults": { "decimals": 4 } }
      },
      {
        "id": 21, "title": "Training Events", "type": "timeseries",
        "gridPos": { "h": 8, "w": 8, "x": 8, "y": 12 },
        "targets": [
          { "expr": "increase(mnemonicai_training_events_total{result=\"success\"}[1h])", "legendFormat": "success" },
          { "expr": "increase(mnemonicai_training_events_total{result=\"failure\"}[1h])", "legendFormat": "failure" },
          { "expr": "increase(mnemonicai_training_events_total{result=\"rollback\"}[1h])", "legendFormat": "rollback" }
        ]
      },
      {
        "id": 22, "title": "Adapter Count", "type": "stat",
        "gridPos": { "h": 8, "w": 8, "x": 16, "y": 12 },
        "targets": [{ "expr": "mnemonicai_adapter_count", "legendFormat": "adapters" }],
        "fieldConfig": { "defaults": { "decimals": 0 } }
      },
      {
        "id": 30, "title": "GPU Memory Usage (GB)", "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 20 },
        "targets": [
          { "expr": "mnemonicai_gpu_memory_used_bytes / 1073741824", "legendFormat": "GPU {{gpu_index}} used" },
          { "expr": "mnemonicai_gpu_memory_total_bytes / 1073741824", "legendFormat": "GPU {{gpu_index}} total" }
        ],
        "fieldConfig": { "defaults": { "unit": "decgbytes", "decimals": 1 } }
      },
      {
        "id": 31, "title": "GPU Utilization (%)", "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 20 },
        "targets": [{ "expr": "mnemonicai_gpu_utilization_percent", "legendFormat": "GPU {{gpu_index}}" }],
        "fieldConfig": { "defaults": { "unit": "percent", "max": 100 } }
      },
      {
        "id": 40, "title": "Health Status", "type": "stat",
        "gridPos": { "h": 4, "w": 24, "x": 0, "y": 28 },
        "targets": [{ "expr": "mnemonicai_health_status", "legendFormat": "health" }],
        "fieldConfig": { "defaults": { "mappings": [{"type":"value","options":{"0":{"text":"UNHEALTHY","color":"red"},"1":{"text":"HEALTHY","color":"green"}}}] } }
      }
    ],
    "templating": { "list": [] },
    "annotations": { "list": [] }
  },
  "overwrite": true
}
EOF

# ============================================================
# grafana/datasource.yaml
# ============================================================
cat > grafana/datasource.yaml <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF

# ============================================================
# grafana/dashboards.yaml
# ============================================================
cat > grafana/dashboards.yaml <<'EOF'
apiVersion: 1
providers:
  - name: "MnemonicAi"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# ============================================================
# docker-compose.observability.yml
# ============================================================
cat > docker-compose.observability.yml <<'EOF'
# docker-compose.observability.yml — Prometheus + Grafana + Node Exporter
# Usage: docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: mnemonicai-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.enable-lifecycle"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: mnemonicai-grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      # Google OAuth
      - GF_AUTH_GOOGLE_ENABLED=true
      - GF_AUTH_GOOGLE_CLIENT_ID=${GRAFANA_GOOGLE_CLIENT_ID}
      - GF_AUTH_GOOGLE_CLIENT_SECRET=${GRAFANA_GOOGLE_CLIENT_SECRET}
      - GF_AUTH_GOOGLE_SCOPES=https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email
      - GF_AUTH_GOOGLE_AUTH_URL=https://accounts.google.com/o/oauth2/auth
      - GF_AUTH_GOOGLE_TOKEN_URL=https://accounts.google.com/o/oauth2/token
      - GF_AUTH_GOOGLE_API_URL=https://www.googleapis.com/oauth2/v1/userinfo
      - GF_AUTH_GOOGLE_ALLOW_SIGN_UP=true
      - GF_AUTH_GOOGLE_HOSTED_DOMAIN=${GRAFANA_HOSTED_DOMAIN:-yourdomain.com}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/datasource.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
      - ./grafana/dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
      - ./grafana/dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: mnemonicai-node-exporter
    ports:
      - "9100:9100"
    pid: host
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - "--path.procfs=/host/proc"
      - "--path.sysfs=/host/sys"
      - "--path.rootfs=/rootfs"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

# ============================================================
# Done
# ============================================================
echo ""
echo "=== All files created ==="
echo ""
find k8s .github app/metrics.py prometheus grafana docker-compose.observability.yml -type f | sort
echo ""
echo "Before deploying:"
echo "  1. Replace YOUR_GITHUB_ORG in k8s/deployment.yaml, k8s/train-job.yaml, k8s/kustomization.yaml"
echo "  2. Add 'prometheus-fastapi-instrumentator>=7.0' to requirements.txt"
echo "  3. Import app/metrics.py in start.py:  from app.metrics import setup_metrics, update_adapter_metrics"
echo "  4. Call setup_metrics(app) after creating the FastAPI app in start.py"
echo ""
echo "Docker Compose observability:"
echo "  docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d"
echo "  Grafana: http://localhost:3000  (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "Kubernetes:"
echo "  kubectl apply -k k8s/"
echo "  kubectl -n mnemonicai get pods"
echo ""
echo "CI/CD:"
echo "  Push to main -> image builds and pushes to ghcr.io automatically"
echo "  Push tag v1.0.0 -> creates GitHub Release + versioned image"
