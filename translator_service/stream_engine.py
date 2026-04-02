"""
Simple per-chunk transcription with initial_prompt context.
No VAD, no rolling buffer, no LocalAgreement.

Routes Georgian audio to the NeMo FastConformer engine and all
other languages through Whisper (mlx-whisper).
"""

from __future__ import annotations

import threading
from typing import Optional

import whisper_engine
import georgian_engine

_PROMPT_TAIL_CHARS = 200


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
        # Route Georgian to the dedicated NeMo engine
        if language == "ka":
            stt = georgian_engine.transcribe(pcm_bytes)
        else:
            # Use Whisper with initial_prompt context for all other languages
            with self._lock:
                prompt = self._last_transcript[-_PROMPT_TAIL_CHARS:] if self._last_transcript else None

            stt = whisper_engine.transcribe(
                pcm_bytes=pcm_bytes,
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


_state: Optional[StreamState] = None
_state_lock = threading.Lock()


def get_state() -> StreamState:
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = StreamState()
    return _state


def warmup_vad() -> None:
    pass  # no VAD
