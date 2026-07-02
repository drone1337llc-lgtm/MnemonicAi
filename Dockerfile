# MnemonicAi — self-hosted ornith-1.0-9b + brain-inspired memory (NVIDIA GPU).
# Mount your model at /models/ornith-1.0-9b and persist memory at /data.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MNEMONICAI_HOST=0.0.0.0 \
    MNEMONICAI_MODEL=/models/ornith-1.0-9b \
    MNEMONICAI_DATA=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install the package with the GPU backend. (Install a CUDA-matched torch first
# if the default wheel doesn't match your driver: see requirements-gpu.txt.)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir ".[gpu]"

VOLUME ["/models", "/data"]
EXPOSE 8400

CMD ["mnemonicai", "serve", "--no-browser"]
