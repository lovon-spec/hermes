"""
Hermes Translator Microservice
FastAPI server on port 5005.

Endpoints
---------
GET  /health     -> {"status": "ready"|"loading"}
POST /translate  -> transcription + translation of raw PCM audio
"""

import asyncio
import logging
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import whisper_engine
import nllb_engine

# ── Logging ────────────────────────────────────────────────────────────
logger = logging.getLogger("hermes.translator")

# ── App state ──────────────────────────────────────────────────────────
_ready = False
app = FastAPI(title="Hermes Translator Service")

# Minimum audio body size: 0.5 s at 16 kHz, 16-bit mono = 16000 bytes/s * 0.5 = 8000 bytes
_MIN_AUDIO_BYTES = 8000


# ── Startup ────────────────────────────────────────────────────────────
def _warmup() -> None:
    """Load both models in a background thread so uvicorn starts accepting
    connections immediately (health returns 'loading' in the meantime)."""
    global _ready
    try:
        logger.info("Warming up Whisper engine...")
        whisper_engine.warmup()
        logger.info("Whisper engine ready.")

        logger.info("Warming up NLLB engine...")
        nllb_engine.warmup()
        logger.info("NLLB engine ready.")

        _ready = True
        logger.info("All models loaded -- service is ready.")
    except Exception:
        logger.exception("Model warmup failed")


@app.on_event("startup")
async def on_startup() -> None:
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> JSONResponse:
    if _ready:
        return JSONResponse({"status": "ready"}, status_code=200)
    return JSONResponse({"status": "loading"}, status_code=503)


# ── Translate ──────────────────────────────────────────────────────────
def _process_audio(pcm_bytes: bytes) -> dict:
    t0 = time.perf_counter()

    stt = whisper_engine.transcribe(pcm_bytes, language=None)
    text = stt["text"]
    source_lang = stt["language"]

    if not text or not text.strip():
        return {
            "translation": "",
            "source_lang": source_lang,
            "original_text": "",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "skipped": True,
        }

    if source_lang == "en":
        return {
            "translation": text,
            "source_lang": "en",
            "original_text": text,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "skipped": True,
        }

    nllb_result = nllb_engine.translate(text, source_lang=source_lang)
    return {
        "translation": nllb_result["translation"],
        "source_lang": source_lang,
        "original_text": text,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "skipped": False,
    }


@app.post("/translate")
async def translate(request: Request) -> JSONResponse:
    if not _ready:
        return JSONResponse(
            {"error": "Service is still loading models. Try again shortly."},
            status_code=503,
        )

    pcm_bytes = await request.body()
    logger.warning(f"Received {len(pcm_bytes)} bytes")

    if len(pcm_bytes) < _MIN_AUDIO_BYTES:
        return JSONResponse({
            "translation": "",
            "source_lang": "",
            "original_text": "",
            "latency_ms": 0,
            "skipped": True,
        })

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _process_audio, pcm_bytes)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Error processing audio")
        return JSONResponse(
            {"error": str(exc)},
            status_code=500,
        )


# ── Parent watchdog ───────────────────────────────────────────────────
def _watch_parent() -> None:
    """Exit if parent process dies (prevents orphaned Python processes)."""
    import os, signal
    ppid = os.getppid()
    while True:
        time.sleep(2)
        try:
            os.kill(ppid, 0)  # check if parent is alive
        except OSError:
            logger.info("Parent process gone, shutting down.")
            os.kill(os.getpid(), signal.SIGTERM)
            break


# ── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=_watch_parent, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=5005, log_level="info")
