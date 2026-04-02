"""
Hermes Translator Microservice
FastAPI server on port 5005.

Endpoints
---------
GET  /health     -> {"status": "ready"|"loading"}
POST /translate  -> streaming transcription + translation of raw PCM audio
POST /reset      -> reset the streaming state for a fresh utterance
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import whisper_engine
import nllb_engine
import stream_engine
import georgian_engine

# -- Logging ---------------------------------------------------------------
logger = logging.getLogger("hermes.translator")

# -- App state -------------------------------------------------------------
_ready = False
app = FastAPI(title="Hermes Translator Service")

# Minimum audio body size: 0.5 s at 16 kHz, 16-bit mono = 16000 bytes/s * 0.5
_MIN_AUDIO_BYTES = 8000


# -- Startup ---------------------------------------------------------------
def _warmup() -> None:
    """Load all models in a background thread so uvicorn starts accepting
    connections immediately (health returns 'loading' in the meantime)."""
    global _ready
    try:
        logger.info("Warming up Whisper engine...")
        whisper_engine.warmup()
        logger.info("Whisper engine ready.")

        logger.info("Warming up Georgian NeMo engine...")
        georgian_engine.warmup()
        logger.info("Georgian NeMo engine ready.")

        logger.info("Warming up NLLB engine...")
        nllb_engine.warmup()
        logger.info("NLLB engine ready.")

        logger.info("Warming up Silero VAD...")
        stream_engine.warmup_vad()
        logger.info("Silero VAD ready.")

        _ready = True
        logger.info("All models loaded -- service is ready.")
    except Exception:
        logger.exception("Model warmup failed")


@app.on_event("startup")
async def on_startup() -> None:
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


# -- Health ----------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    if _ready:
        return JSONResponse({"status": "ready"}, status_code=200)
    return JSONResponse({"status": "loading"}, status_code=503)


# -- Reset -----------------------------------------------------------------
@app.post("/reset")
async def reset() -> JSONResponse:
    """Clear the streaming buffer and transcription state."""
    stream_engine.get_state().reset()
    return JSONResponse({"status": "ok"})


# -- Translate -------------------------------------------------------------
def _process_audio(pcm_bytes: bytes, language: Optional[str]) -> dict:
    """Run per-chunk transcription with VAD + initial_prompt, then NLLB."""
    t0 = time.perf_counter()

    state = stream_engine.get_state()
    result = state.process_chunk(pcm_bytes, language=language)

    text = result["text"]
    source_lang = result["source_lang"]

    # Strip punctuation-only transcriptions (NeMo outputs "." for silence)
    import re
    cleaned = re.sub(r'[^\w]', '', text or '')
    if not cleaned:
        return {
            "translation": "",
            "original_text": "",
            "tentative": "",
            "source_lang": source_lang,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "is_final": False,
            "skipped": True,
        }

    # Translate through NLLB if non-English
    translation = text
    skipped = True
    if source_lang != "en":
        nllb_result = nllb_engine.translate(text, source_lang=source_lang)
        translation = nllb_result["translation"]
        skipped = False

    return {
        "translation": translation,
        "original_text": text,
        "tentative": "",
        "source_lang": source_lang,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "is_final": True,
        "skipped": skipped,
    }


@app.post("/translate")
async def translate(request: Request) -> JSONResponse:
    if not _ready:
        return JSONResponse(
            {"error": "Service is still loading models. Try again shortly."},
            status_code=503,
        )

    # Read optional language header (default: auto-detect)
    lang_header = request.headers.get("X-Language", "auto")
    language: Optional[str] = None if lang_header == "auto" else lang_header

    pcm_bytes = await request.body()
    logger.info("Received %d bytes (lang=%s)", len(pcm_bytes), lang_header)

    if len(pcm_bytes) < _MIN_AUDIO_BYTES:
        return JSONResponse({
            "translation": "",
            "original_text": "",
            "tentative": "",
            "source_lang": "",
            "latency_ms": 0,
            "is_final": False,
            "skipped": True,
        })

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _process_audio, pcm_bytes, language)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Error processing audio")
        return JSONResponse(
            {"error": str(exc)},
            status_code=500,
        )


# -- Parent watchdog -------------------------------------------------------
def _watch_parent() -> None:
    """Exit if parent process dies (prevents orphaned Python processes)."""
    import os
    import signal
    ppid = os.getppid()
    while True:
        time.sleep(2)
        current_ppid = os.getppid()
        # If reparented to launchd (PID 1) or original parent is gone, exit
        if current_ppid == 1 or current_ppid != ppid:
            logger.info("Parent gone (ppid changed %d -> %d), shutting down.", ppid, current_ppid)
            os._exit(0)  # hard exit, don't wait for stuck GPU ops
            break
        try:
            os.kill(ppid, 0)
        except OSError:
            logger.info("Parent process gone, shutting down.")
            os._exit(0)
            break


# -- Entrypoint ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=_watch_parent, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=5005, log_level="info")
