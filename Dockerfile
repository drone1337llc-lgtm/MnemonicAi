# Dockerfile — MnemonicAi inference + training image
# Target: CUDA 12.6 + Python 3.12
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-developer

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

# ---------- system deps ----------
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl wget build-essential cmake pkg-config \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- venv ----------
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel setuptools "setuptools<70"

# ---------- PyTorch (CUDA 12.6) ----------
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# ---------- GPU training stack ----------
RUN pip install --no-cache-dir \
    "bitsandbytes>=0.45" \
    "accelerate>=1.0" \
    "peft>=0.14" \
    "transformers>=4.45" \
    "safetensors>=0.5" \
    "sentencepiece>=0.2" \
    "datasets>=2.20"

# ---------- llama-cpp-python (CUDA) ----------
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall \
    "llama-cpp-python[cuda]"

# ---------- app deps ----------
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "pydantic>=2.10" \
    "httpx>=0.28" \
    "sse-starlette>=2.1" \
    "python-multipart>=0.0.18" \
    "prometheus-fastapi-instrumentator>=7.0" \
    "prometheus-client>=0.21"

# ---------- app copy ----------
WORKDIR /
COPY src/ .

# ---------- editable install ----------
RUN pip install runpod transformers

# ---------- entrypoint ----------
RUN chmod +x /app/mn_*.sh 2>/dev/null || true

EXPOSE 8400 8401

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8400/health || exit 1

CMD ["python", "-u", "/handler.py"]
