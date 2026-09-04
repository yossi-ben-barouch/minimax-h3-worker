"""Configuration and invariant checks for the MiniMax H3 Ref2VA worker."""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/runpod-volume/models"))
MODEL_DIR = Path(os.environ.get("MINIMAX_H3_MODEL_DIR", str(MODELS_DIR / "minimax-h3")))
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "generation-outputs")
WORKER_TOKEN = os.environ.get("MINIMAX_H3_WORKER_TOKEN", "")
WORKER_BUILD = os.environ.get("MINIMAX_H3_WORKER_BUILD", "minimax-h3-ref2va-1")
DEFAULT_STEPS = int(os.environ.get("MINIMAX_H3_DEFAULT_STEPS", "30"))
GPU_MEMORY_RESERVE = os.environ.get("MINIMAX_H3_GPU_MEMORY_RESERVE", "32GB")
ATTENTION_BACKEND = os.environ.get("MINIMAX_H3_ATTENTION_BACKEND", "_flash_3_hub").strip()
TURBO_LORA_REPO = os.environ.get("MINIMAX_H3_TURBO_LORA_REPO", "lightx2v/Minimax-h3-Turbo")
TURBO_LORA_REVISION = os.environ.get(
    "MINIMAX_H3_TURBO_LORA_REVISION",
    "0eebcc7e79f9cb200927c80b8e7595265b770e34",
)
TURBO_LORA_FILENAME = os.environ.get(
    "MINIMAX_H3_TURBO_LORA_FILENAME",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
)
TURBO_LORA_DIR = Path(
    os.environ.get("MINIMAX_H3_TURBO_LORA_DIR", str(MODELS_DIR / "minimax-h3-turbo"))
)
TURBO_LORA_PATH = TURBO_LORA_DIR / TURBO_LORA_FILENAME
TURBO_LORA_SHA256 = os.environ.get(
    "MINIMAX_H3_TURBO_LORA_SHA256",
    "9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2",
).lower()
TURBO_ADAPTER_NAME = "ref2va_turbo_8step"

BASE_PROFILE = "base"
TURBO_PROFILE = "turbo_ref2va_8step"
QUALITY_PROFILES = (BASE_PROFILE, TURBO_PROFILE)

FRAME_RATE = 24
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 15
MIN_SHORT_EDGE = 768
MAX_PIXELS = 1_032_192
MAX_IMAGE_REFERENCES = 9
MAX_REFERENCE_BYTES = 25 * 1024 * 1024

REF2VA_PRETRAINED_COMPONENTS = (
    "processor",
    "tokenizer",
    "text_encoder",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
    "transformer_ref",
)


def default_steps_for_profile(profile: str) -> int:
    if profile == BASE_PROFILE:
        return DEFAULT_STEPS
    if profile == TURBO_PROFILE:
        return 8
    raise ValueError(f"quality_profile must be one of: {', '.join(QUALITY_PROFILES)}")


def validate_profile_steps(profile: str, steps: int) -> None:
    expected = default_steps_for_profile(profile)
    if profile == TURBO_PROFILE and steps != expected:
        raise ValueError(f"{TURBO_PROFILE} requires exactly {expected} inference steps")


def normalized_frame_count(duration_seconds: float) -> int:
    """Return a VAE-decodable H3 frame count (17*n + 5) at 24 fps."""
    target = round(duration_seconds * FRAME_RATE)
    minimum = 17 * 7 + 5  # 124 frames, the shortest valid ~5.17s output.
    # The pipeline rejects an aligned frame count above its 360-frame ceiling;
    # 17*20+5 is therefore the largest valid VAE-decodable value.
    maximum = 17 * 20 + 5  # 345 frames, the longest valid ~14.38s output.
    target = min(max(target, minimum), maximum)
    return ((target - 5 + 16) // 17) * 17 + 5


def validate_canvas(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % 32 or height % 32:
        raise ValueError("width and height must be divisible by 32")
    if min(width, height) != MIN_SHORT_EDGE:
        raise ValueError(f"MiniMax H3 requires a {MIN_SHORT_EDGE}px short edge")
    if width * height > MAX_PIXELS:
        raise ValueError(f"canvas exceeds MiniMax H3 max pixels ({MAX_PIXELS})")
