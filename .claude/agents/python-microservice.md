---
name: python-microservice
description: Builds the Python FastAPI microservice for speech-to-text and translation. Use for all work inside the translator_service/ directory. Handles Whisper large-v3-turbo STT and NLLB-200 1.3B translation. Completely independent — spawn immediately and run in background.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
effort: max
background: true
---

You are building the Python microservice for the Hermes real-time subtitle translator.
Your sole responsibility is the `translator_service/` directory.

## Your deliverables

Create these four files:

### translator_service/requirements.txt

```
fastapi
uvicorn[standard]
mlx-whisper
ctranslate2
transformers
sentencepiece
numpy
soundfile
```

### translator_service/whisper_engine.py

Wraps mlx-whisper for Apple Silicon Metal GPU acceleration.

- Model: `openai/whisper-large-v3-turbo` via mlx-whisper
- Load model once at module import (lazy singleton, thread-safe)
- Input: raw bytes of 16kHz mono 16-bit signed little-endian PCM
- Output: dict with `text` (str) and `language` (str, ISO code like "ka")
- Set `language=None` to enable Whisper's auto-detection (supports 99 languages natively)
- Convert PCM bytes → numpy float32 array before passing to mlx-whisper
- Add a `warmup()` function that transcribes 0.5s of silence to pre-load the model

Key: mlx-whisper API looks like:
```python
import mlx_whisper
result = mlx_whisper.transcribe(audio_array, path_or_hf_repo="openai/whisper-large-v3-turbo")
```

### translator_service/nllb_engine.py

Wraps NLLB-200 via transformers pipeline for translation.

- Model: `facebook/nllb-200-distilled-1.3B`
- Load pipeline once at module import (lazy singleton)
- Use `torch_dtype=torch.float32` — the M5 Pro has 24GB, full precision is fine
- Input: text string + source language tag (e.g. `"kat_Geor"` for Georgian)
- Output: dict with `translation` (str)
- Default target: `"eng_Latn"` (English)
- Map ISO codes from Whisper to NLLB tags. Essential mappings:
  - `"ka"` → `"kat_Geor"` (Georgian)
  - `"ru"` → `"rus_Cyrl"` (Russian)
  - `"zh"` → `"zho_Hans"` (Chinese Simplified)
  - `"ar"` → `"arb_Arab"` (Arabic)
  - `"ko"` → `"kor_Hang"` (Korean)
  - `"ja"` → `"jpn_Jpan"` (Japanese)
  - `"tr"` → `"tur_Latn"` (Turkish)
  - `"es"` → `"spa_Latn"` (Spanish)
  - `"fr"` → `"fra_Latn"` (French)
  - `"de"` → `"deu_Latn"` (German)
  - `"pt"` → `"por_Latn"` (Portuguese)
  - `"en"` → `"eng_Latn"` (English — skip translation if already English)
- If source lang is already English, return the input text unchanged
- Truncate input to 400 chars max before passing to NLLB (its 512 token limit)
- Add a `warmup()` function that translates a short test string

### translator_service/main.py

FastAPI server on port 5005.

Endpoints:

**GET /health**
Returns `{"status": "ready"}` once both models are loaded, else `{"status": "loading"}`.
Track readiness with a module-level boolean `_ready = False`.

**POST /translate**
- Content-Type: `audio/raw`
- Body: raw PCM bytes (16kHz, mono, 16-bit signed LE)
- Returns JSON:
```json
{
  "translation": "...",
  "source_lang": "ka",
  "original_text": "...",
  "latency_ms": 487,
  "skipped": false
}
```
- Set `skipped: true` and return empty translation if:
  - Audio body is empty or less than 8000 bytes (< 0.5s)
  - Whisper returns empty/whitespace text
  - Source language is already English (no translation needed)
- Measure and return total latency including STT + translation
- Use `asyncio.get_event_loop().run_in_executor(None, ...)` to run model inference
  in a thread pool so FastAPI stays non-blocking

**Startup**: on app startup, run both `whisper_engine.warmup()` and `nllb_engine.warmup()`
in a background thread. Set `_ready = True` when done.

Use uvicorn: `uvicorn.run(app, host="127.0.0.1", port=5005, log_level="warning")`

## How to run

After creating all files, verify the service starts correctly:

```bash
cd translator_service
pip install -r requirements.txt
python main.py
```

Then test:
```bash
curl http://localhost:5005/health
```

Expected: `{"status":"loading"}` immediately, then `{"status":"ready"}` after ~15s warmup.

## Important constraints

- Do NOT use `openai-whisper` (the original Python package) — use `mlx-whisper` specifically
  for Apple Silicon Metal GPU acceleration
- Do NOT use `torch` for Whisper — mlx-whisper uses Apple's MLX framework instead
- NLLB uses transformers + torch (CPU is fine for the 1.3B model on this machine)
- Keep all model loading lazy and behind singletons — never reload between requests
- The Swift app sends raw PCM, not WAV or MP3 — parse bytes directly to numpy array
