from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiResponse:
    ok: bool
    message: str


def has_api_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key.strip())


def analyze_image_bytes(_image_bytes: bytes) -> GeminiResponse:
    if not has_api_key():
        return GeminiResponse(ok=False, message="GEMINI_API_KEY no configurada")

    raise NotImplementedError("Gemini API no está conectada aún. Configura GEMINI_API_KEY y habilita la implementación.")
