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
import os
import threading
import time
from typing import Optional

import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import whisper_engine
import nllb_engine
import stream_engine
import georgian_engine

# -- Logging ---------------------------------------------------------------
logger = logging.getLogger("hermes.translator")

# -- Google Translate -------------------------------------------------------
_GOOGLE_TRANSLATE_KEY = ""
# Check env vars first
for env_name in ["GOOGLE_TRANSLATE", "TRANSLATE_API_KEY"]:
    _GOOGLE_TRANSLATE_KEY = os.environ.get(env_name, "")
    if _GOOGLE_TRANSLATE_KEY:
        break
# Fall back to ~/.env file
if not _GOOGLE_TRANSLATE_KEY:
    _env_path = os.path.expanduser("~/.env")
    if os.path.isfile(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                for key_name in ["GOOGLE_TRANSLATE", "TRANSLATE_API_KEY"]:
                    if line.startswith(key_name + "="):
                        _GOOGLE_TRANSLATE_KEY = line.split("=", 1)[1]
                        break
                if _GOOGLE_TRANSLATE_KEY:
                    break


def _google_translate(text: str, source_lang: str = "ka", target_lang: str = "en") -> str:
    """Translate text using Google Cloud Translation API v2."""
    if not _GOOGLE_TRANSLATE_KEY:
        return ""
    try:
        resp = http_requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            params={"key": _GOOGLE_TRANSLATE_KEY},
            json={"q": text, "source": source_lang, "target": target_lang, "format": "text"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()["data"]["translations"][0]["translatedText"]
        logger.warning("Google Translate HTTP %d: %s", resp.status_code, resp.text[:100])
        return ""
    except Exception as e:
        logger.warning("Google Translate error: %s", e)
        return ""


def _google_stt(pcm_bytes: bytes, language: str = "ka-GE") -> str:
    """Transcribe audio using Google Cloud Speech-to-Text v1."""
    if not _GOOGLE_TRANSLATE_KEY:
        return ""
    import base64
    audio_b64 = base64.b64encode(pcm_bytes).decode()
    try:
        resp = http_requests.post(
            "https://speech.googleapis.com/v1/speech:recognize",
            params={"key": _GOOGLE_TRANSLATE_KEY},
            json={
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": 16000,
                    "languageCode": language,
                    "model": "default",
                    "enableAutomaticPunctuation": True,
                },
                "audio": {"content": audio_b64},
            },
            timeout=10,
        )
        logger.warning("Google STT raw HTTP %d, body len=%d", resp.status_code, len(resp.text))
        data = resp.json()
        if resp.status_code == 200:
            results = data.get("results", [])
            if results:
                transcript = results[0]["alternatives"][0]["transcript"]
                logger.warning("Google STT transcript: %s", transcript[:100])
                return transcript
            return ""
        logger.warning("Google STT error response: %s", str(data)[:200])
        return ""
    except Exception as e:
        logger.warning("Google STT exception: %s", e)
        return ""

# -- Gemini API key --------------------------------------------------------
_GEMINI_KEY = ""
for env_name in ["GEMINI_API_KEY"]:
    _GEMINI_KEY = os.environ.get(env_name, "")
    if _GEMINI_KEY:
        break
if not _GEMINI_KEY:
    _env_path = os.path.expanduser("~/.env")
    if os.path.isfile(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    _GEMINI_KEY = line.split("=", 1)[1]
                    break


def _gemini_translate_audio(pcm_bytes: bytes) -> tuple:
    """Send audio to Gemini Flash, get Georgian transcription + English translation.
    Returns (translation, original_text) or ("", "") on failure."""
    if not _GEMINI_KEY:
        return "", ""
    import base64
    import wave
    import io

    # Convert PCM to WAV in memory (Gemini needs a proper audio format)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    wav_bytes = wav_buf.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()

    try:
        resp = http_requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-live-preview:generateContent?key={_GEMINI_KEY}",
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                        {"text": "Listen to this audio. If there is Georgian speech, transcribe it in Georgian script and translate it to English. Reply with ONLY two lines:\nLine 1: Georgian transcription\nLine 2: English translation\nIf there is no speech or you cannot understand it, reply with just: EMPTY"},
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
            },
            timeout=15,
        )
        logger.warning("Gemini HTTP %d", resp.status_code)
        if resp.status_code != 200:
            logger.warning("Gemini error: %s", resp.text[:200])
            return "", ""

        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.warning("Gemini response: %s", text[:150])

        if text == "EMPTY" or not text:
            return "", ""

        lines = text.split("\n", 1)
        if len(lines) >= 2:
            return lines[1].strip(), lines[0].strip()  # (translation, original)
        return lines[0].strip(), ""  # just translation
    except Exception as e:
        logger.warning("Gemini error: %s", e)
        return "", ""


# -- App state -------------------------------------------------------------
app = FastAPI(title="Hermes Translator Service")

# Minimum audio body size: 0.5 s at 16 kHz, 16-bit mono = 16000 bytes/s * 0.5
_MIN_AUDIO_BYTES = 8000


# -- Startup ---------------------------------------------------------------
# Service is ready immediately — local models load lazily on first use
_ready = True


@app.on_event("startup")
async def on_startup() -> None:
    pass  # no eager warmup — models load on demand


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
    """Route audio through local or cloud pipeline based on language."""
    t0 = time.perf_counter()
    import re

    # Gemini pipeline: audio → Georgian transcription + English translation in one call
    if language == "ka-gemini":
        translation, original = _gemini_translate_audio(pcm_bytes)
        if not translation:
            return _empty_result("ka", t0)
        return {
            "translation": translation,
            "original_text": original,
            "tentative": "",
            "source_lang": "ka",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "is_final": True,
            "skipped": False,
        }

    # Cloud pipeline: Google STT + Google Translate
    if language == "ka-cloud":
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        logger.warning("Cloud pipeline: sending %d bytes to Google STT (RMS=%.0f)", len(pcm_bytes), rms)
        text = _google_stt(pcm_bytes, language="ka-GE")
        logger.warning("Cloud STT returned: %s", (text or "")[:80])
        if not text or not text.strip():
            return _empty_result("ka", t0)
        translation = _google_translate(text, source_lang="ka")
        logger.warning("Cloud Translate returned: %s", (translation or "")[:80])
        return {
            "translation": translation or text,
            "original_text": text,
            "tentative": "",
            "source_lang": "ka",
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "is_final": True,
            "skipped": not translation,
        }

    # Local pipeline: NeMo/Whisper + Google Translate/NLLB
    state = stream_engine.get_state()
    result = state.process_chunk(pcm_bytes, language=language)

    text = result["text"]
    source_lang = result["source_lang"]

    cleaned = re.sub(r'[^\w]', '', text or '')
    if len(cleaned) < 5:
        return _empty_result(source_lang, t0)

    # Translate if non-English
    translation = text
    skipped = True
    if source_lang != "en":
        translation = _google_translate(text, source_lang)
        if not translation:
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


def _empty_result(source_lang: str, t0: float) -> dict:
    return {
        "translation": "",
        "original_text": "",
        "tentative": "",
        "source_lang": source_lang,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "is_final": False,
        "skipped": True,
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
    logger.warning("Received %d bytes (lang=%s)", len(pcm_bytes), lang_header)

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
