"""No-GPU contract tests for the MiniMax H3 worker."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main() -> None:
    assert config.GPU_MEMORY_RESERVE.endswith("GB")
    assert config.ATTENTION_BACKEND in {"", "_flash_3_hub"}
    assert config.TURBO_LORA_SHA256 == "9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2"
    assert config.default_steps_for_profile(config.BASE_PROFILE) == config.DEFAULT_STEPS
    assert config.default_steps_for_profile(config.TURBO_PROFILE) == 8
    config.validate_profile_steps(config.TURBO_PROFILE, 8)
    try:
        config.validate_profile_steps(config.TURBO_PROFILE, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("Turbo profile unexpectedly accepted a non-8-step request")
    assert "transformer_ref" in config.REF2VA_PRETRAINED_COMPONENTS
    assert "transformer" not in config.REF2VA_PRETRAINED_COMPONENTS
    assert config.normalized_frame_count(5) == 124
    assert config.normalized_frame_count(14) == 345
    assert config.normalized_frame_count(15) == 345
    config.validate_canvas(768, 1344)
    config.validate_canvas(768, 768)
    config.validate_canvas(1344, 768)
    for width, height in ((704, 1280), (768, 1376), (769, 1344)):
        try:
            config.validate_canvas(width, height)
        except ValueError:
            continue
        raise AssertionError(f"invalid canvas unexpectedly passed: {width}x{height}")
    print("MiniMax H3 worker contract smoke test passed")


if __name__ == "__main__":
    main()
