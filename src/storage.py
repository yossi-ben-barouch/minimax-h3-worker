"""Minimal Supabase Storage uploader used by the GPU worker."""
from __future__ import annotations

import os
from urllib.parse import quote

import requests

from . import config


def _headers(content_type: str) -> dict[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not service_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": content_type,
        "x-upsert": "true",
        "Cache-Control": "public, max-age=86400",
    }


def upload_bytes(path: str, payload: bytes, content_type: str) -> str:
    base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not base_url:
        raise RuntimeError("SUPABASE_URL is required")
    encoded_path = quote(path, safe="/")
    response = requests.post(
        f"{base_url}/storage/v1/object/{config.OUTPUT_BUCKET}/{encoded_path}",
        headers=_headers(content_type),
        data=payload,
        timeout=120,
    )
    response.raise_for_status()
    return path
