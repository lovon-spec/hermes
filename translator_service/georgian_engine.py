"""
Georgian STT engine using NVIDIA NeMo FastConformer.

Model: nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc
  - 115M parameters, ~460MB
  - 7.44% WER on Georgian (vs ~35% for Whisper)
  - FastConformer architecture with cache-aware streaming
  - Runs on CPU (no CUDA/MPS on macOS)

Input:  raw PCM bytes (16 kHz, mono, 16-bit signed little-endian)
Output: dict with 'text' (str) and 'language' (str, always "ka")
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import wave

import numpy as np

logger = logging.getLogger("hermes.georgian")

_MODEL_NAME = "nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc"
_SAMPLE_RATE = 16000

_lock = threading.Lock()
_model = None

# Serialize inference calls -- the model is not thread-safe
_inference_lock = threading.Lock()


def _get_model():
    """Return the NeMo ASR model, loading it exactly once (lazy singleton)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading NeMo Georgian model: %s", _MODEL_NAME)

                # NeMo needs these environment settings before import
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

                import torch
                # Use all available CPU cores for inference
                torch.set_num_threads(os.cpu_count() or 8)
                torch.set_num_interop_threads(4)
                # Suppress noisy NeMo/PL logs during loading
                logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
                logging.getLogger("nemo_logger").setLevel(logging.WARNING)

                import nemo.collections.asr as nemo_asr

                model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                    model_name=_MODEL_NAME,
                )
                model = model.to("cpu")
                model.eval()

                # Disable decoding strategies that might try to use CUDA
                model.change_decoding_strategy(decoder_type="ctc")

                _model = model
                logger.info("NeMo Georgian model loaded successfully.")
    return _model


def _pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Convert raw 16-bit signed LE PCM bytes to float32 numpy array."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def _write_temp_wav(audio: np.ndarray) -> str:
    """Write float32 audio array to a temporary WAV file. Returns the path."""
    # Convert back to int16 for WAV file
    int16_audio = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(int16_audio.tobytes())
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    else:
        os.close(fd)
    return path


def transcribe(pcm_bytes: bytes) -> dict:
    """
    Transcribe raw PCM audio bytes to Georgian text.

    Parameters
    ----------
    pcm_bytes : bytes
        Raw 16 kHz, mono, 16-bit signed little-endian PCM audio.

    Returns
    -------
    dict  {"text": str, "language": "ka"}
    """
    if not pcm_bytes or len(pcm_bytes) < 1600:  # less than 0.05s
        return {"text": "", "language": "ka"}

    audio = _pcm_bytes_to_float32(pcm_bytes)

    if len(audio) == 0:
        return {"text": "", "language": "ka"}

    # NeMo's transcribe() expects file paths, so write a temp WAV
    wav_path = _write_temp_wav(audio)
    try:
        model = _get_model()

        import torch

        with _inference_lock:
            with torch.no_grad():
                # NeMo transcribe returns a list of transcription strings
                results = model.transcribe([wav_path])

                # Handle different NeMo return formats:
                # Some versions return list of strings, others return
                # a Hypothesis object or a tuple (hypotheses, _)
                if isinstance(results, tuple):
                    results = results[0]

                if results and len(results) > 0:
                    text = results[0]
                    # If it is a Hypothesis object, extract the text
                    if hasattr(text, "text"):
                        text = text.text
                    text = str(text).strip()
                else:
                    text = ""

    finally:
        # Clean up temp file
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    return {"text": text, "language": "ka"}


def warmup() -> None:
    """Pre-load the model by transcribing a short silence."""
    logger.info("Warming up Georgian NeMo engine...")
    silence = np.zeros(8000, dtype=np.float32)  # 0.5s of silence

    wav_path = _write_temp_wav(silence)
    try:
        model = _get_model()

        import torch

        with _inference_lock:
            with torch.no_grad():
                model.transcribe([wav_path])
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    logger.info("Georgian NeMo engine ready.")
