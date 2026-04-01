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

# Use the mlx-community repo which ships weights in MLX format
# (weights.safetensors + native config keys).  The openai/ repo uses
# PyTorch format which mlx_whisper cannot load directly.
_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"

# Lazy singleton -- the first call to _get_module() triggers the import,
# which in turn downloads / caches the model weights via mlx-whisper.
_lock = threading.Lock()
_mlx_whisper = None


def _get_module():
    """Return the mlx_whisper module, importing it exactly once (thread-safe)."""
    global _mlx_whisper
    if _mlx_whisper is None:
        with _lock:
            if _mlx_whisper is None:
                import mlx_whisper  # noqa: F811 -- heavy import, done once
                _mlx_whisper = mlx_whisper
    return _mlx_whisper


def _pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit signed LE PCM bytes to a float32 numpy array in [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def transcribe(pcm_bytes: bytes, *, language: Optional[str] = None) -> dict:
    """
    Transcribe raw PCM audio bytes.

    Parameters
    ----------
    pcm_bytes : bytes
        16 kHz, mono, 16-bit signed little-endian PCM.
    language : str or None
        ISO 639-1 code (e.g. "ka" for Georgian).  Pass None for auto-detection.

    Returns
    -------
    dict  {"text": str, "language": str}
    """
    mlx_w = _get_module()
    audio = _pcm_bytes_to_float32(pcm_bytes)

    kwargs = {
        "path_or_hf_repo": _MODEL_REPO,
    }
    if language is not None:
        kwargs["language"] = language

    result = mlx_w.transcribe(audio, **kwargs)

    text = (result.get("text") or "").strip()
    detected_lang = result.get("language", language or "")

    return {
        "text": text,
        "language": detected_lang,
    }


def warmup() -> None:
    """Pre-load the model by transcribing 0.5 s of silence."""
    silence = np.zeros(8000, dtype=np.float32)  # 0.5 s at 16 kHz
    mlx_w = _get_module()
    mlx_w.transcribe(silence, path_or_hf_repo=_MODEL_REPO)
