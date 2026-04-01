"""
Whisper STT engine using mlx-whisper for Apple Silicon Metal GPU acceleration.

Model: mlx-community/whisper-large-v3-turbo (MLX-format weights for Metal GPU)
       This is the MLX-converted version of openai/whisper-large-v3-turbo.
Input:  raw bytes of 16 kHz mono 16-bit signed little-endian PCM
Output: dict with 'text' (str) and 'language' (str, ISO 639-1 code)
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"

_lock = threading.Lock()
_mlx_whisper = None

# Serialize all MLX/Metal GPU operations to prevent concurrent command buffer access
_inference_lock = threading.Lock()

# Silence threshold: RMS below this means no meaningful audio
_SILENCE_RMS_THRESHOLD = 0.01


def _get_module():
    """Return the mlx_whisper module, importing it exactly once (thread-safe)."""
    global _mlx_whisper
    if _mlx_whisper is None:
        with _lock:
            if _mlx_whisper is None:
                import mlx_whisper
                _mlx_whisper = mlx_whisper
    return _mlx_whisper


def _pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit signed LE PCM bytes to a float32 numpy array in [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def _is_silence(audio: np.ndarray) -> bool:
    """Return True if the audio is effectively silent."""
    if len(audio) == 0:
        return True
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < _SILENCE_RMS_THRESHOLD


def transcribe(pcm_bytes: bytes, *, language: Optional[str] = None) -> dict:
    """
    Transcribe raw PCM audio bytes.

    Returns dict {"text": str, "language": str}
    """
    audio = _pcm_bytes_to_float32(pcm_bytes)

    if _is_silence(audio):
        return {"text": "", "language": language or ""}

    mlx_w = _get_module()

    kwargs = {
        "path_or_hf_repo": _MODEL_REPO,
    }
    if language is not None:
        kwargs["language"] = language

    # Serialize Metal GPU access — concurrent MLX calls crash the command buffer
    with _inference_lock:
        result = mlx_w.transcribe(audio, **kwargs)

    text = (result.get("text") or "").strip()
    detected_lang = result.get("language", language or "")

    return {
        "text": text,
        "language": detected_lang,
    }


def warmup() -> None:
    """Pre-load the model by transcribing 0.5 s of silence."""
    silence = np.zeros(8000, dtype=np.float32)
    mlx_w = _get_module()
    with _inference_lock:
        mlx_w.transcribe(silence, path_or_hf_repo=_MODEL_REPO)
