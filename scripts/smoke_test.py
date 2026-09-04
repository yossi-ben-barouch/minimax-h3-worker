"""No-GPU contract tests for the MiniMax H3 worker."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def main() -> None:
    assert config.normalized_frame_count(5) == 124
    assert config.normalized_frame_count(15) == 362
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
