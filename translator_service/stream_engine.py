"""
Streaming transcription engine with rolling buffer and LocalAgreement-2.

Maintains a rolling audio buffer (max 30 s at 16 kHz).  Each incoming chunk
is VAD-filtered (Silero), appended to the buffer, and then the entire buffer
is transcribed.  Results are stabilised via LocalAgreement-2: only text that
two consecutive transcriptions agree on is emitted as "confirmed".  Everything
after the agreed prefix is "tentative".
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

import whisper_engine

logger = logging.getLogger("hermes.stream")

# ── Constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE = 16_000
_MAX_BUFFER_SECONDS = 10
_MAX_BUFFER_SAMPLES = _SAMPLE_RATE * _MAX_BUFFER_SECONDS
_PROMPT_TAIL_CHARS = 200  # trailing chars of confirmed text used as initial_prompt


# ── Silero VAD wrapper ───────────────────────────────────────────────────

_vad_lock = threading.Lock()
_vad_model = None


def _get_vad():
    """Return the Silero VAD model, loading it exactly once (thread-safe)."""
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
    """Return True if Silero VAD detects speech in *audio* (16 kHz float32)."""
    import torch
    model, _utils = _get_vad()
    # Silero expects a 1-D float32 torch tensor, sample rate 16000
    tensor = torch.from_numpy(audio).float()
    # Process in 512-sample (32 ms) windows; if any window exceeds the
    # threshold we consider the chunk as containing speech.
    window = 512
    for start in range(0, len(tensor), window):
        segment = tensor[start : start + window]
        if len(segment) < window:
            # Pad the last segment if it is shorter than the window size
            segment = torch.nn.functional.pad(segment, (0, window - len(segment)))
        confidence = model(segment, _SAMPLE_RATE).item()
        if confidence > threshold:
            return True
    return False


def _detect_speech_end(audio: np.ndarray, tail_seconds: float = 0.8,
                       threshold: float = 0.15) -> bool:
    """Return True if the last *tail_seconds* of *audio* contain no speech,
    indicating the speaker has stopped (useful for finalising an utterance)."""
    tail_samples = int(_SAMPLE_RATE * tail_seconds)
    if len(audio) < tail_samples:
        return False
    tail = audio[-tail_samples:]
    return not _chunk_has_speech(tail, threshold=threshold)


def warmup_vad() -> None:
    """Pre-load the Silero VAD model by running inference on silence."""
    _get_vad()
    silence = np.zeros(8000, dtype=np.float32)
    _chunk_has_speech(silence)
    logger.info("Silero VAD warmed up.")


# ── LocalAgreement-2 ─────────────────────────────────────────────────────

def _local_agreement(previous: str, current: str) -> tuple[str, str]:
    """Compare *previous* and *current* transcription and return
    ``(confirmed, tentative)`` based on their longest common prefix,
    computed at the word level.

    ``confirmed`` is the word-level prefix that both transcriptions agree on.
    ``tentative`` is the remainder of *current* that is not yet confirmed.
    """
    prev_words = previous.split()
    curr_words = current.split()

    # Find the length of the longest common word-level prefix
    match_len = 0
    for pw, cw in zip(prev_words, curr_words):
        if pw == cw:
            match_len += 1
        else:
            break

    confirmed = " ".join(curr_words[:match_len])
    tentative = " ".join(curr_words[match_len:])
    return confirmed, tentative


# ── StreamState ──────────────────────────────────────────────────────────

class StreamState:
    """Per-session rolling buffer and transcription state.

    This class is the main entry point used by ``main.py``.  A single global
    instance is created and reused across all requests (the service handles
    one audio source at a time).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._prev_full_text: str = ""      # last full transcription
        self._confirmed_text: str = ""      # all confirmed text so far
        self._last_emitted: str = ""        # confirmed text already sent to the client
        self._detected_lang: str = ""

    def reset(self) -> None:
        """Clear all state to start a fresh utterance."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._prev_full_text = ""
            self._confirmed_text = ""
            self._last_emitted = ""
            self._detected_lang = ""

    # ── public API ───────────────────────────────────────────────────
    def process_chunk(
        self,
        pcm_bytes: bytes,
        language: Optional[str] = None,
    ) -> dict:
        """Append *pcm_bytes* to the rolling buffer, run VAD + Whisper +
        LocalAgreement, and return a result dict.

        Parameters
        ----------
        pcm_bytes : bytes
            Raw 16 kHz mono 16-bit signed LE PCM.
        language : str | None
            ISO 639-1 code or ``None`` for auto-detect.

        Returns
        -------
        dict with keys: confirmed, tentative, source_lang, latency_ms,
                        is_final
        """
        t0 = time.perf_counter()

        audio_chunk = whisper_engine._pcm_bytes_to_float32(pcm_bytes)

        # ---- VAD gate ------------------------------------------------
        if not _chunk_has_speech(audio_chunk):
            # No speech in this chunk.
            # Check whether we should finalise an ongoing utterance.
            with self._lock:
                if len(self._buffer) > 0 and self._confirmed_text:
                    # Speech ended -- finalise
                    return self._finalise(t0)
            return self._empty_result(t0)

        # ---- Append to rolling buffer --------------------------------
        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Trim front if buffer exceeds maximum length
            if len(self._buffer) > _MAX_BUFFER_SAMPLES:
                excess = len(self._buffer) - _MAX_BUFFER_SAMPLES
                self._buffer = self._buffer[excess:]

            buffer_copy = self._buffer.copy()
            prompt = self._confirmed_text[-_PROMPT_TAIL_CHARS:] if self._confirmed_text else None

        # ---- Transcribe the full buffer (outside the state lock) -----
        stt = whisper_engine.transcribe(
            audio_array=buffer_copy,
            language=language,
            initial_prompt=prompt,
        )
        current_text = stt["text"]
        detected_lang = stt["language"]

        # ---- LocalAgreement-2 ----------------------------------------
        with self._lock:
            self._detected_lang = detected_lang

            if not current_text.strip():
                return self._empty_result(t0)

            confirmed, tentative = _local_agreement(
                self._prev_full_text, current_text
            )
            self._prev_full_text = current_text

            # Accumulate confirmed text
            if confirmed:
                # Only append the *new* confirmed words (those beyond what was
                # already in self._confirmed_text).
                prev_confirmed_words = self._confirmed_text.split()
                confirmed_words = confirmed.split()
                if len(confirmed_words) > len(prev_confirmed_words):
                    new_words = confirmed_words[len(prev_confirmed_words):]
                    if self._confirmed_text:
                        self._confirmed_text += " " + " ".join(new_words)
                    else:
                        self._confirmed_text = " ".join(new_words)

            # Determine newly confirmed text (not yet sent to client)
            newly_confirmed = ""
            if self._confirmed_text and self._confirmed_text != self._last_emitted:
                if self._last_emitted and self._confirmed_text.startswith(self._last_emitted):
                    newly_confirmed = self._confirmed_text[len(self._last_emitted):].strip()
                else:
                    newly_confirmed = self._confirmed_text
                self._last_emitted = self._confirmed_text

            # Check for speech end (finalize utterance)
            is_final = False
            if len(buffer_copy) > _SAMPLE_RATE and _detect_speech_end(buffer_copy):
                is_final = True
                final_text = (self._confirmed_text + " " + tentative).strip() if tentative else self._confirmed_text
                newly_confirmed = final_text
                if self._last_emitted and final_text.startswith(self._last_emitted):
                    newly_confirmed = final_text[len(self._last_emitted):].strip()
                self._last_emitted = final_text
                # Reset state for next utterance but keep the language
                saved_lang = self._detected_lang
                self._buffer = np.zeros(0, dtype=np.float32)
                self._prev_full_text = ""
                self._confirmed_text = ""
                self._last_emitted = ""
                self._detected_lang = saved_lang

            latency = int((time.perf_counter() - t0) * 1000)
            return {
                "confirmed": newly_confirmed,
                "tentative": tentative if not is_final else "",
                "source_lang": self._detected_lang,
                "latency_ms": latency,
                "is_final": is_final,
            }

    # ── internal helpers ─────────────────────────────────────────────
    def _finalise(self, t0: float) -> dict:
        """Mark the current confirmed text as final and reset."""
        remaining = self._confirmed_text
        if self._last_emitted and remaining.startswith(self._last_emitted):
            remaining = remaining[len(self._last_emitted):].strip()
        lang = self._detected_lang
        # Reset for next utterance
        self._buffer = np.zeros(0, dtype=np.float32)
        self._prev_full_text = ""
        self._confirmed_text = ""
        self._last_emitted = ""
        latency = int((time.perf_counter() - t0) * 1000)
        return {
            "confirmed": remaining,
            "tentative": "",
            "source_lang": lang,
            "latency_ms": latency,
            "is_final": True,
        }

    def _empty_result(self, t0: float) -> dict:
        latency = int((time.perf_counter() - t0) * 1000)
        return {
            "confirmed": "",
            "tentative": "",
            "source_lang": self._detected_lang,
            "latency_ms": latency,
            "is_final": False,
        }


# ── Module-level singleton ───────────────────────────────────────────────
_state: Optional[StreamState] = None
_state_lock = threading.Lock()


def get_state() -> StreamState:
    """Return the global StreamState singleton."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = StreamState()
    return _state
