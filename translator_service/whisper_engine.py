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


def transcribe(
    pcm_bytes: Optional[bytes] = None,
    *,
    audio_array: Optional[np.ndarray] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Transcribe raw PCM audio bytes or a pre-built float32 numpy array.

    Pass *either* ``pcm_bytes`` (raw 16-bit LE PCM) **or** ``audio_array``
    (float32 numpy, values in [-1, 1]).  If both are given, ``audio_array``
    takes precedence.

    Parameters
    ----------
    pcm_bytes : bytes | None
        Raw 16 kHz mono 16-bit signed little-endian PCM.
    audio_array : np.ndarray | None
        Float32 numpy array already converted, values in [-1, 1].
    language : str | None
        ISO 639-1 language code for Whisper.  ``None`` = auto-detect.
    initial_prompt : str | None
        Prompt text to condition the decoder for better continuity.

    Returns dict {"text": str, "language": str}
    """
    if audio_array is not None:
        audio = audio_array
    elif pcm_bytes is not None:
        audio = _pcm_bytes_to_float32(pcm_bytes)
    else:
        return {"text": "", "language": language or ""}

    if len(audio) == 0:
        return {"text": "", "language": language or ""}

    mlx_w = _get_module()

    kwargs = {
        "path_or_hf_repo": _MODEL_REPO,
        "condition_on_previous_text": False,  # prevents hallucination loops
        "compression_ratio_threshold": 1.8,   # reject highly repetitive output
        "no_speech_threshold": 0.5,           # skip low-confidence segments
    }
    if language is not None:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    # Serialize Metal GPU access — concurrent MLX calls crash the command buffer
    with _inference_lock:
        result = mlx_w.transcribe(audio, **kwargs)

    text = (result.get("text") or "").strip()

    # Detect and reject hallucination loops (repeated characters/words)
    if len(text) > 20:
        words = text.split()
        if len(words) > 3 and len(set(words)) <= 2:
            text = ""  # all same word repeated = hallucination
        # Check for repeated character patterns
        unique_chars = set(text.replace(" ", ""))
        if len(unique_chars) <= 3 and len(text) > 10:
            text = ""  # e.g. "ეეეეეეეეეეე"
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
