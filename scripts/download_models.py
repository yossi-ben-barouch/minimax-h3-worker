"""Stage only the MiniMax H3 Ref2VA partition and shared components on a Network Volume."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/runpod-volume/models"))
TARGET = Path(os.environ.get("MINIMAX_H3_MODEL_DIR", str(MODELS_DIR / "minimax-h3")))

# The root diffusers repository includes both task-family transformers. Ref2VA
# only needs transformer_ref plus the shared conditioner, VAEs, schedulers, and
# tokenizer/processor files. Excluding transformer/ avoids downloading FL2VA.
ALLOW_PATTERNS = [
    "model_index.json",
    "modular_model_index.json",
    "processor/**",
    "tokenizer/**",
    "text_encoder/**",
    "vae/**",
    "audio_vae/**",
    "scheduler/**",
    "audio_scheduler/**",
    "transformer_ref/**",
]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="MiniMaxAI/MiniMax-H3",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(TARGET),
        token=os.environ.get("HF_TOKEN") or None,
    )
    required = [
        TARGET / "modular_model_index.json",
        *(TARGET / subfolder for subfolder in config.REF2VA_PRETRAINED_COMPONENTS),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing staged MiniMax H3 files: {', '.join(map(str, missing))}")

    config.TURBO_LORA_DIR.mkdir(parents=True, exist_ok=True)
    turbo_path = Path(
        hf_hub_download(
            repo_id=config.TURBO_LORA_REPO,
            filename=config.TURBO_LORA_FILENAME,
            revision=config.TURBO_LORA_REVISION,
            local_dir=str(config.TURBO_LORA_DIR),
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    turbo_digest = _sha256(turbo_path)
    if turbo_digest != config.TURBO_LORA_SHA256:
        raise SystemExit(
            f"MiniMax H3 Turbo LoRA checksum mismatch: expected {config.TURBO_LORA_SHA256}, got {turbo_digest}"
        )

    total = sum(path.stat().st_size for path in TARGET.rglob("*") if path.is_file())
    print(f"MiniMax H3 Ref2VA ready at {TARGET} ({total / 1e9:.1f} GB)")
    print(f"MiniMax H3 Ref2VA Turbo LoRA ready at {turbo_path} ({turbo_path.stat().st_size / 1e9:.2f} GB)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
