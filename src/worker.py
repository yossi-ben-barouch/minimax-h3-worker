"""Reference-to-audio-video inference using the official Diffusers H3 pipeline."""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.utils.export_utils import encode_video
from PIL import Image

from . import config

logger = logging.getLogger("minimax-h3-worker")


class MiniMaxH3Worker:
    """Owns one lazily loaded Ref2VA pipeline for a Serverless container."""

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._turbo_adapter_loaded = False
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        with self._load_lock:
            if self._pipe is not None:
                return self._pipe
            if not config.MODEL_DIR.joinpath("modular_model_index.json").exists():
                raise RuntimeError(
                    f"MiniMax H3 model is not staged at {config.MODEL_DIR}. "
                    "Run scripts/download_models.py on the attached Network Volume first."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for MiniMax H3 inference")

            logger.info("loading MiniMax H3 Ref2VA from %s", config.MODEL_DIR)
            manager = ComponentsManager()
            pipe = ModularPipeline.from_pretrained(
                str(config.MODEL_DIR),
                workflow="ref2va",
                components_manager=manager,
                local_files_only=True,
            )
            # The published modular index points each component back to the Hub
            # repository. Override that path so a cold worker uses only the
            # already-staged Network Volume and never downloads 100+ GB again.
            pipe.load_components(
                dtype=torch.bfloat16,
                pretrained_model_name_or_path=str(config.MODEL_DIR),
                local_files_only=True,
            )
            missing = [
                name
                for name in config.REF2VA_PRETRAINED_COMPONENTS
                if getattr(pipe, name, None) is None
            ]
            if missing:
                raise RuntimeError(
                    "MiniMax H3 failed to load required local components: "
                    f"{', '.join(missing)}. Check the preceding component-loader warnings."
                )
            if config.TURBO_LORA_PATH.exists():
                turbo_digest = self._sha256(config.TURBO_LORA_PATH)
                if turbo_digest != config.TURBO_LORA_SHA256:
                    raise RuntimeError(
                        "MiniMax H3 Turbo LoRA checksum mismatch: "
                        f"expected {config.TURBO_LORA_SHA256}, got {turbo_digest}"
                    )
                pipe.load_lora_weights(
                    str(config.TURBO_LORA_DIR),
                    weight_name=config.TURBO_LORA_FILENAME,
                    adapter_name=config.TURBO_ADAPTER_NAME,
                    load_into_transformer_ref=True,
                )
                pipe.disable_lora()
                self._turbo_adapter_loaded = True
                logger.info(
                    "loaded MiniMax H3 Ref2VA Turbo adapter %s",
                    config.TURBO_LORA_PATH,
                )
            else:
                logger.warning(
                    "MiniMax H3 Ref2VA Turbo adapter is not staged at %s; base profile remains available",
                    config.TURBO_LORA_PATH,
                )
            # Attach hooks after registration so every model participates in
            # eviction. A 32 GB reserve prevents a 141 GB H200 from retaining
            # both 62 GB giants and starving transformer activations.
            manager.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin=config.GPU_MEMORY_RESERVE,
            )
            if config.ATTENTION_BACKEND:
                pipe.transformer_ref.set_attention_backend(config.ATTENTION_BACKEND)
            self._pipe = pipe
            logger.info(
                "MiniMax H3 Ref2VA loaded from the Network Volume on %s with %s reserve and %s attention (%s)",
                torch.cuda.get_device_name(0),
                config.GPU_MEMORY_RESERVE,
                config.ATTENTION_BACKEND or "native",
                ", ".join(config.REF2VA_PRETRAINED_COMPONENTS),
            )
            return pipe

    def load_diagnostics(self) -> dict[str, Any]:
        """Load the model without generating media for a paid deployment check."""
        pipe = self._ensure_loaded()
        return {
            "loaded": True,
            "model_dir": str(config.MODEL_DIR),
            "workflow": "ref2va",
            "components": [
                name
                for name in config.REF2VA_PRETRAINED_COMPONENTS
                if getattr(pipe, name, None) is not None
            ],
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_reserve": config.GPU_MEMORY_RESERVE,
            "attention_backend": config.ATTENTION_BACKEND or "native",
            "worker_build": config.WORKER_BUILD,
            "quality_profiles": [config.BASE_PROFILE]
            + ([config.TURBO_PROFILE] if self._turbo_adapter_loaded else []),
            "turbo_lora": {
                "loaded": self._turbo_adapter_loaded,
                "repo": config.TURBO_LORA_REPO,
                "revision": config.TURBO_LORA_REVISION,
                "filename": config.TURBO_LORA_FILENAME,
            },
        }

    def generate(
        self,
        *,
        prompt: str,
        reference_image_urls: list[str],
        duration_seconds: float,
        width: int,
        height: int,
        num_inference_steps: int,
        quality_profile: str,
        seed: int | None,
        enable_audio: bool,
    ) -> tuple[bytes, bytes, dict[str, Any]]:
        if not prompt.strip():
            raise ValueError("prompt is required")
        if not reference_image_urls:
            raise ValueError("at least one reference image is required")
        if len(reference_image_urls) > config.MAX_IMAGE_REFERENCES:
            raise ValueError(f"at most {config.MAX_IMAGE_REFERENCES} image references are supported")
        if not config.MIN_DURATION_SECONDS <= duration_seconds <= config.MAX_DURATION_SECONDS:
            raise ValueError("duration_seconds must be between 5 and 15")
        if not 1 <= num_inference_steps <= 80:
            raise ValueError("num_inference_steps must be between 1 and 80")
        config.validate_profile_steps(quality_profile, num_inference_steps)
        config.validate_canvas(width, height)

        request_started = time.perf_counter()
        pipe = self._ensure_loaded()
        with tempfile.TemporaryDirectory(prefix="minimax-h3-") as temp_dir, self._generate_lock:
            root = Path(temp_dir)
            references_started = time.perf_counter()
            references = [
                MiniMaxH3ImageReference.from_file(str(self._download_reference(url, root, index)))
                for index, url in enumerate(reference_image_urls, start=1)
            ]
            references_seconds = time.perf_counter() - references_started
            frames = config.normalized_frame_count(duration_seconds)
            generator = torch.Generator(device="cuda")
            if seed is not None:
                generator.manual_seed(seed)
                effective_seed = seed
            else:
                effective_seed = generator.seed()

            if quality_profile == config.TURBO_PROFILE:
                if not self._turbo_adapter_loaded:
                    raise RuntimeError(
                        "the Ref2VA Turbo adapter is not staged; run admin_action=stage_models and restart the worker"
                    )
                pipe.set_adapters(config.TURBO_ADAPTER_NAME, adapter_weights=1.0)
                pipe.enable_lora()
            else:
                if self._turbo_adapter_loaded:
                    pipe.disable_lora()

            logger.info(
                "starting Ref2VA inference: profile=%s, refs=%s, frames=%s, canvas=%sx%s, steps=%s, seed=%s",
                quality_profile,
                len(references),
                frames,
                width,
                height,
                num_inference_steps,
                effective_seed,
            )
            inference_started = time.perf_counter()
            results = pipe(
                prompt=prompt,
                references=references,
                num_frames=frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output=["videos", "audio", "sampling_rate"],
            )
            inference_seconds = time.perf_counter() - inference_started

            encode_started = time.perf_counter()
            video_path = root / "output.mp4"
            video_frames = results["videos"][0]
            if enable_audio:
                encode_video(
                    video_frames,
                    fps=config.FRAME_RATE,
                    output_path=str(video_path),
                    audio=results["audio"][0],
                    audio_sample_rate=results["sampling_rate"],
                )
            else:
                encode_video(video_frames, fps=config.FRAME_RATE, output_path=str(video_path))

            thumbnail_path = root / "thumbnail.jpg"
            self._write_thumbnail(video_frames[0], thumbnail_path)
            encode_seconds = time.perf_counter() - encode_started
            diag = {
                "worker_build": config.WORKER_BUILD,
                "model": "MiniMaxAI/MiniMax-H3",
                "workflow": "ref2va",
                "references": len(references),
                "frame_rate": config.FRAME_RATE,
                "num_frames": frames,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "quality_profile": quality_profile,
                "audio": enable_audio,
                "seed": effective_seed,
                "seed_was_generated": seed is None,
                "gpu": torch.cuda.get_device_name(0),
                "attention_backend": config.ATTENTION_BACKEND or "native",
                "timings_seconds": {
                    "reference_download_and_decode": round(references_seconds, 3),
                    "pipeline_inference": round(inference_seconds, 3),
                    "encode_and_thumbnail": round(encode_seconds, 3),
                    "worker_generate_total": round(time.perf_counter() - request_started, 3),
                },
            }
            return video_path.read_bytes(), thumbnail_path.read_bytes(), diag

    @staticmethod
    def _download_reference(url: str, destination: Path, index: int) -> Path:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"reference {index} is not an image")
        suffix = ".png" if "png" in content_type else ".jpg"
        path = destination / f"reference-{index}{suffix}"
        written = 0
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > config.MAX_REFERENCE_BYTES:
                    raise ValueError(f"reference {index} exceeds the 25 MB limit")
                file.write(chunk)
        return path

    @staticmethod
    def _write_thumbnail(frame: Any, destination: Path) -> None:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().float().cpu().numpy()
        image = np.asarray(frame)
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        Image.fromarray(image).convert("RGB").save(destination, format="JPEG", quality=90)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
