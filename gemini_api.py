from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiResponse:
    ok: bool
    message: str
    data: dict | None = None


def has_api_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key.strip())


def calls_enabled() -> bool:
    return os.environ.get("GEMINI_ENABLE_CALLS", "0").strip() == "1"


def _post_generate_content(*, model: str, api_key: str, payload: dict) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def ruler_endpoints_from_image_bytes(image_bytes: bytes, *, model: str = "gemini-1.5-flash") -> GeminiResponse:
    """Attempts to extract the two endpoints of the 30cm ruler in pixel coordinates using Gemini.

    Safety: This function will NOT call the API unless GEMINI_ENABLE_CALLS=1.
    """

    if not has_api_key():
        return GeminiResponse(ok=False, message="GEMINI_API_KEY no configurada")

    if not calls_enabled():
        return GeminiResponse(
            ok=False,
            message="Llamadas a Gemini deshabilitadas. Para habilitar: export GEMINI_ENABLE_CALLS=1",
        )

    api_key = os.environ["GEMINI_API_KEY"].strip()
    b64 = base64.b64encode(image_bytes).decode("ascii")

    prompt = (
        "Analiza la imagen. Hay una regla de 30 cm visible completa. "
        "Devuelve SOLO JSON estricto con el formato: "
        "{\"ruler\": {\"p1\": {\"x\": <int>, \"y\": <int>}, \"p2\": {\"x\": <int>, \"y\": <int>}}}. "
        "p1 y p2 deben ser los extremos de la regla (0 cm y 30 cm) en coordenadas de pixel. "
        "No incluyas texto adicional."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        out = _post_generate_content(model=model, api_key=api_key, payload=payload)
    except Exception as e:
        return GeminiResponse(ok=False, message=f"Error llamando Gemini: {e}")

    try:
        text = out["candidates"][0]["content"]["parts"][0].get("text", "")
        data = json.loads(text)
    except Exception:
        return GeminiResponse(ok=False, message="Respuesta inválida de Gemini (no JSON)")

    ruler = data.get("ruler")
    if not isinstance(ruler, dict):
        return GeminiResponse(ok=False, message="JSON sin campo 'ruler'")

    return GeminiResponse(ok=True, message="ok", data=data)
