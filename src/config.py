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

FRAME_RATE = 24
MIN_DURATION_SECONDS = 5
MAX_DURATION_SECONDS = 15
MIN_SHORT_EDGE = 768
MAX_PIXELS = 1_032_192
MAX_IMAGE_REFERENCES = 9
MAX_REFERENCE_BYTES = 25 * 1024 * 1024


def normalized_frame_count(duration_seconds: float) -> int:
    """Return a VAE-decodable H3 frame count (17*n + 5) at 24 fps."""
    target = round(duration_seconds * FRAME_RATE)
    minimum = 17 * 7 + 5  # 124 frames, the shortest valid ~5.17s output.
    maximum = 17 * 21 + 5  # 362 frames, the longest valid ~15.08s output.
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
