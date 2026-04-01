"""
Simple per-chunk transcription with VAD gating and initial_prompt context.

Each chunk is transcribed independently. The previous transcription is passed
as initial_prompt for continuity. Silero VAD skips silence to save GPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

import whisper_engine

logger = logging.getLogger("hermes.stream")

_SAMPLE_RATE = 16_000
_PROMPT_TAIL_CHARS = 200

# ── Silero VAD ───────────────────────────────────────────────────────────

_vad_lock = threading.Lock()
_vad_model = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                _vad_model = (model, utils)
    return _vad_model


def _chunk_has_speech(audio: np.ndarray, threshold: float = 0.35) -> bool:
    import torch
    model, _ = _get_vad()
    tensor = torch.from_numpy(audio).float()
    # Sample 3 evenly-spaced windows instead of scanning every 512 samples
    window = 512
    length = len(tensor)
    if length < window:
        return False
    positions = [0, length // 2, length - window]
    for start in positions:
        segment = tensor[start : start + window]
        if model(segment, _SAMPLE_RATE).item() > threshold:
            return True
    return False


def warmup_vad() -> None:
    _get_vad()
    _chunk_has_speech(np.zeros(8000, dtype=np.float32))
    logger.info("Silero VAD warmed up.")


# ── Transcription state ──────────────────────────────────────────────────

class StreamState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_transcript: str = ""
        self._detected_lang: str = ""

    def reset(self) -> None:
        with self._lock:
            self._last_transcript = ""
            self._detected_lang = ""

    def process_chunk(self, pcm_bytes: bytes, language: Optional[str] = None) -> dict:
        audio = whisper_engine._pcm_bytes_to_float32(pcm_bytes)

        # VAD: skip silence
        if not _chunk_has_speech(audio):
            return {
                "text": "",
                "source_lang": self._detected_lang,
                "is_final": False,
            }

        # Get context from previous transcription
        with self._lock:
            prompt = self._last_transcript[-_PROMPT_TAIL_CHARS:] if self._last_transcript else None

        # Transcribe this chunk independently
        stt = whisper_engine.transcribe(
            audio_array=audio,
            language=language,
            initial_prompt=prompt,
        )

        text = stt["text"]
        detected_lang = stt["language"]

        with self._lock:
            self._detected_lang = detected_lang
            if text.strip():
                self._last_transcript = text

        return {
            "text": text,
            "source_lang": detected_lang,
            "is_final": True,
        }


# ── Singleton ────────────────────────────────────────────────────────────

_state: Optional[StreamState] = None
_state_lock = threading.Lock()


def get_state() -> StreamState:
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = StreamState()
    return _state
