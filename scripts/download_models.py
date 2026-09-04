"""Stage only the MiniMax H3 Ref2VA partition and shared components on a Network Volume."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

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
    required = [TARGET / "modular_model_index.json", TARGET / "transformer_ref", TARGET / "text_encoder"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing staged MiniMax H3 files: {', '.join(map(str, missing))}")
    total = sum(path.stat().st_size for path in TARGET.rglob("*") if path.is_file())
    print(f"MiniMax H3 Ref2VA ready at {TARGET} ({total / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
