"""
Whisper STT engine using mlx-whisper for Apple Silicon Metal GPU acceleration.

Uses turbo model for English (fast, ~900ms) and full large-v3 for other
languages like Georgian (slower but accurate, no hallucination loops).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

# Turbo for English/auto — fast, 4 decoder layers
_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"
# Full large-v3 for non-English — 32 decoder layers, much better for low-resource langs
_MODEL_FULL = "mlx-community/whisper-large-v3-mlx"

_lock = threading.Lock()
_mlx_whisper = None

# Serialize all MLX/Metal GPU operations to prevent concurrent command buffer access
_inference_lock = threading.Lock()


def _get_module():
    global _mlx_whisper
    if _mlx_whisper is None:
        with _lock:
            if _mlx_whisper is None:
                import mlx_whisper
                _mlx_whisper = mlx_whisper
    return _mlx_whisper


def _pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def transcribe(
    pcm_bytes: Optional[bytes] = None,
    *,
    audio_array: Optional[np.ndarray] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    if audio_array is not None:
        audio = audio_array
    elif pcm_bytes is not None:
        audio = _pcm_bytes_to_float32(pcm_bytes)
    else:
        return {"text": "", "language": language or ""}

    if len(audio) == 0:
        return {"text": "", "language": language or ""}

    mlx_w = _get_module()

    # Use full model for non-English, turbo for English/auto
    if language is not None and language != "en":
        model = _MODEL_FULL
    else:
        model = _MODEL_TURBO

    kwargs = {
        "path_or_hf_repo": model,
    }
    if language is not None:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    with _inference_lock:
        result = mlx_w.transcribe(audio, **kwargs)

    text = (result.get("text") or "").strip()
    detected_lang = result.get("language", language or "")

    return {
        "text": text,
        "language": detected_lang,
    }


def warmup() -> None:
    """Pre-load the turbo model (full model loads on first non-English request)."""
    silence = np.zeros(8000, dtype=np.float32)
    mlx_w = _get_module()
    with _inference_lock:
        mlx_w.transcribe(silence, path_or_hf_repo=_MODEL_TURBO)
