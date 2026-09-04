"""RunPod Serverless entry point for MiniMax H3 Base Ref2VA."""
from __future__ import annotations

import hmac
import logging
import subprocess
import sys
import traceback
from typing import Any
from uuid import UUID

import runpod

from . import config
from .storage import upload_bytes
from .worker import MiniMaxH3Worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimax-h3-handler")
WORKER = MiniMaxH3Worker()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job = event.get("input") or {}
    try:
        _require_token(job)
        if job.get("admin_action") == "stage_models":
            return _stage_models()
        if job.get("admin_action") == "validate_components":
            return WORKER.load_diagnostics()
        job_id = _uuid_string(job, "_job_id")
        user_id = _uuid_string(job, "_user_id")
        prompt = _string(job, "prompt")
        reference_urls = _reference_urls(job)
        duration = _number(job, "duration_seconds", 5)
        width = int(_number(job, "width", 768))
        height = int(_number(job, "height", 1344))
        steps = int(_number(job, "num_inference_steps", config.DEFAULT_STEPS))
        seed = job.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer")
        enable_audio = job.get("enable_audio", True)
        if not isinstance(enable_audio, bool):
            raise ValueError("enable_audio must be a boolean")

        video, thumbnail, diag = WORKER.generate(
            prompt=prompt,
            reference_image_urls=reference_urls,
            duration_seconds=duration,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed,
            enable_audio=enable_audio,
        )
        storage_path = upload_bytes(f"{user_id}/{job_id}.mp4", video, "video/mp4")
        thumbnail_path = upload_bytes(f"{user_id}/{job_id}.jpg", thumbnail, "image/jpeg")
        diag.update({"bytes": len(video), "thumbnail_bytes": len(thumbnail)})
        return {"storage_path": storage_path, "thumbnail_path": thumbnail_path, "diag": diag}
    except PermissionError as error:
        return {"error": str(error), "error_type": type(error).__name__}
    except Exception as error:
        logger.exception("MiniMax H3 job failed")
        return {
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }


def _stage_models() -> dict[str, Any]:
    """Hydrate the persistent model volume without loading the pipeline."""
    logger.info("Starting MiniMax H3 model-volume staging")
    subprocess.run(
        [sys.executable, "scripts/download_models.py"],
        check=True,
        timeout=60 * 60,
    )
    return {"staged": True, "models_dir": str(config.MODELS_DIR)}


def _require_token(job: dict[str, Any]) -> None:
    if not config.WORKER_TOKEN:
        raise RuntimeError("MINIMAX_H3_WORKER_TOKEN is not configured")
    provided = job.get("worker_token")
    if not isinstance(provided, str) or not hmac.compare_digest(provided, config.WORKER_TOKEN):
        raise PermissionError("missing or invalid worker_token")


def _string(job: dict[str, Any], key: str) -> str:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _number(job: dict[str, Any], key: str, default: float) -> float:
    value = job.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _uuid_string(job: dict[str, Any], key: str) -> str:
    value = _string(job, key)
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{key} must be a UUID") from error
    if parsed.int == 0:
        raise ValueError(f"{key} must not be the nil UUID")
    return str(parsed)


def _reference_urls(job: dict[str, Any]) -> list[str]:
    raw = job.get("reference_image_urls")
    if raw is None and isinstance(job.get("reference_image_url"), str):
        raw = [job["reference_image_url"]]
    if not isinstance(raw, list) or not raw:
        raise ValueError("reference_image_urls must contain at least one image URL")
    urls = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    if len(urls) != len(raw):
        raise ValueError("reference_image_urls must contain only non-empty strings")
    return urls


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
