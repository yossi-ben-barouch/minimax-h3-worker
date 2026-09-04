# MiniMax H3 Base Ref2VA RunPod Serverless worker.
#
# The 120+ GB model weights intentionally live on a RunPod Network Volume at
# /runpod-volume/models/minimax-h3 rather than in this image. That makes scale
# to zero practical and keeps the image independently deployable.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/runpod-volume/.cache/huggingface \
    MODELS_DIR=/runpod-volume/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ffmpeg git ca-certificates libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m venv "$VIRTUAL_ENV"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# MiniMax H3 is supported by Diffusers' modular pipeline. Pinning a verified
# source commit makes model behavior reproducible while still using the official
# implementation instead of an unmaintained ComfyUI repack.
ARG DIFFUSERS_REF=7643c4826609c47755e3da0e5b768e8070468f49
RUN python -m pip install "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REF}"

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY src ./src
COPY scripts ./scripts

# Proves that the image contains the H3 pipeline classes before it reaches
# RunPod. No model weights are loaded during build.
RUN python scripts/smoke_test.py \
    && python -c "import torch; assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 9), torch.__version__; from diffusers import ComponentsManager, ModularPipeline; from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference; print('MiniMax H3 runtime imports OK on torch', torch.__version__)"

CMD ["python", "-m", "src.handler"]
