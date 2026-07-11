# Dockerfile — MnemonicAi inference + training image
# Target: CUDA 12.6 + Python 3.12
FROM runpod/pytorch:1.0.7-rc.138-cu1281-torch260-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

# ---------- system deps ----------
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl wget rsync build-essential cmake pkg-config ninja-build \
    libssl-dev libffi-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- llama.cpp (llama-server binary + GGUF conversion scripts) ----------
# Matches the live hybrid-backend architecture: inference runs through a
# compiled llama-server BINARY (blue/green pair, managed by hotswap.py via
# subprocess), not llama-cpp-python bindings. The base image's own CUDA
# driver library (libcuda.so.1) isn't present at build time (it's only
# injected at container run time by nvidia-container-toolkit), so linking
# the final llama-server executable needs --allow-shlib-undefined to skip
# ld's stricter transitive symbol check; the real driver satisfies those
# symbols lazily once the container actually runs.
# Architectures: 80=A100, 86=Ampere (3090/A6000), 89=Ada (4090/L40),
# 90=Hopper (H100) — covers the common RunPod serverless GPU pool.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -B /opt/llama.cpp/build -S /opt/llama.cpp \
        -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build /opt/llama.cpp/build --target llama-server llama-quantize \
        -j"$(nproc)"

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
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir runpod

# ---------- editable install ----------
RUN pip install -e /app 2>/dev/null || true

# ---------- entrypoint ----------
RUN chmod +x /app/mn_*.sh 2>/dev/null || true

VOLUME ["/models", "/data"]
EXPOSE 8400 8401

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8400/health || exit 1

# RunPod serverless entrypoint — see .runpod/handler.py
CMD [ "python3", "-u", ".runpod/handler.py" ]
