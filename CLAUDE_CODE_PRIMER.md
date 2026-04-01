# Project Primer: Real-Time Subtitle Translator for iPhone Mirroring
> Hand this file to Claude Code as your starting context.

---

## What We're Building

A **macOS menu bar app** that listens to audio coming from **iPhone Mirroring** and displays real-time translated subtitles in a floating overlay window — like subtitles on top of TikTok videos, for languages TikTok doesn't support (primarily Georgian).

Everything runs **100% locally and for free** — no API keys, no internet required after setup.

---

## Hardware & Environment

- **Machine:** MacBook Pro with Apple M5 Pro (15-core CPU, 16-core GPU, 24GB unified RAM)
- **OS:** macOS (Sequoia or later required for iPhone Mirroring)
- **Languages:** Swift for the macOS app, Python for the AI microservice

---

## Architecture Overview

```
iPhone (any app, e.g. TikTok)
        │
        ▼  [iPhone Mirroring — routes audio through Mac audio system]
macOS Audio System
        │
        ▼  [ScreenCaptureKit — captures audio from iPhone Mirroring window only]
Swift App (menu bar + overlay)
        │   Sends raw PCM audio chunks via local socket/HTTP
        ▼
Python Microservice (localhost:5005)
        │
        ├─▶ WhisperKit / whisper.cpp  →  Georgian (or other) text
        │
        └─▶ NLLB-200 1.3B distilled  →  English translation
                │
                ▼
        Returns translated text to Swift app
                │
                ▼
Floating NSPanel overlay  (always-on-top subtitle window, positioned over iPhone Mirroring)
```

---

## Component Breakdown

### 1. Swift macOS App

**Responsibilities:**
- Menu bar icon to start/stop translation
- ScreenCaptureKit audio capture (targets iPhone Mirroring window specifically)
- Sends PCM audio chunks (~2 seconds each) to Python microservice via HTTP POST to `localhost:5005/translate`
- Receives translated text and displays it in a floating `NSPanel` subtitle overlay

**Key APIs:**
- `ScreenCaptureKit` (`SCStream` with `SCStreamConfiguration` — set `capturesAudio = true`)
- `SCShareableContent` to find and target the "iPhone Mirroring" window/app
- `NSPanel` with `NSFloatingWindowLevel` for always-on-top overlay
- `URLSession` for local HTTP to the Python service

**Important ScreenCaptureKit notes:**
- Requires `com.apple.security.screen-recording` entitlement
- User must grant Screen Recording permission once in System Preferences
- Use `SCContentFilter(desktopIndependentWindow:)` to isolate iPhone Mirroring audio only, so the Mac's own audio is not captured

**Audio chunk strategy:**
- Capture 2-second PCM buffers at 16kHz, mono (Whisper's required format)
- Buffer and overlap slightly (0.5s) to avoid cutting words at chunk boundaries
- Convert `CMSampleBuffer` → raw PCM bytes → send as `audio/raw` body to microservice

---

### 2. Python Microservice

**Responsibilities:**
- Receives raw PCM audio from Swift app
- Runs Whisper STT → gets Georgian (or auto-detected) transcript
- Runs NLLB-200 translation → gets English text
- Returns JSON `{"translation": "...", "source_lang": "...", "latency_ms": ...}`

**Stack:**
- `Flask` or `FastAPI` for the HTTP server (FastAPI preferred for async)
- `mlx-whisper` or `whisper.cpp` Python bindings for STT
- `transformers` + `ctranslate2` for NLLB-200

**Recommended setup:**

```bash
pip install fastapi uvicorn mlx-whisper ctranslate2 transformers sentencepiece
```

**Whisper model:** `openai/whisper-large-v3-turbo`
- Use `mlx-whisper` for Apple Silicon Metal GPU acceleration
- Set `language="ka"` for Georgian explicitly (faster, more accurate than auto-detect)
- Or pass `language=None` to auto-detect (for multi-language support)

**NLLB-200 model:** `facebook/nllb-200-distilled-1.3B`
- Run at full FP32 (24GB RAM makes this trivial)
- Source lang tag for Georgian: `"kat_Geor"`
- Target lang tag for English: `"eng_Latn"`
- Load model once at startup and keep in memory (do NOT reload per request)

**Microservice endpoint:**

```python
POST /translate
Content-Type: audio/raw
Body: raw 16kHz mono PCM bytes (16-bit signed little-endian)

Response:
{
  "translation": "Here is the English text",
  "source_lang": "ka",        # ISO code detected by Whisper
  "confidence": 0.97,
  "latency_ms": 480
}
```

---

## Model Details

### Whisper large-v3-turbo
- **Parameters:** ~809M (1B range)
- **RAM usage:** ~1.6GB FP16
- **Speed on M5 Pro:** ~0.3s for a 2s audio chunk via MLX/Metal
- **Georgian WER:** <10% (high accuracy tier)
- **Auto-detects language:** Yes — Whisper identifies the language before transcribing
- **Note:** Turbo variant has 4 decoder layers (vs 32 in large-v3), making it 5x faster with near-identical accuracy for transcription. Do NOT use `--task translate` flag with turbo — only transcribe, then use NLLB for translation.

### NLLB-200 distilled 1.3B
- **Parameters:** 1.3B
- **RAM usage:** ~2.6GB FP32
- **Speed on M5 Pro:** ~0.15–0.2s per sentence
- **Why not 3.3B?** The 3.3B dense model falls back to CPU on most setups and is much slower, with only marginal quality gain for Georgian. Distilled 1.3B is the correct choice.
- **Why not Google Translate?** NLLB-200 was specifically trained on low-resource languages including Georgian (`kat_Geor`) and outperforms Google Translate for this language pair in many real-world cases.

### Total resource usage
- RAM: ~5–6GB of 24GB (leaving 18GB completely free)
- Expected end-to-end latency: **~0.5 seconds** from audio chunk end to subtitle display

---

## Overlay UI Design

- Frameless `NSPanel` window, `NSFloatingWindowLevel`
- Positioned at the bottom of the iPhone Mirroring window (dynamically track window position)
- Dark semi-transparent background, white text — classic subtitle look
- Text appears, stays for ~3 seconds, fades out if no new text arrives
- Font: SF Pro or system font, ~18pt, with drop shadow for readability
- Width matches iPhone Mirroring window width, height auto-adjusts to text

---

## Project File Structure (Suggested)

```
TikTokTranslator/
├── TikTokTranslator.xcodeproj
├── TikTokTranslator/
│   ├── AppDelegate.swift          # Menu bar setup
│   ├── AudioCaptureManager.swift  # ScreenCaptureKit logic
│   ├── TranslationClient.swift    # HTTP client to Python service
│   ├── SubtitleOverlayWindow.swift # NSPanel floating overlay
│   └── Info.plist                 # Screen recording entitlement
└── translator_service/
    ├── main.py                    # FastAPI server
    ├── whisper_engine.py          # Whisper STT wrapper
    ├── nllb_engine.py             # NLLB translation wrapper
    └── requirements.txt
```

---

## Key Constraints & Gotchas

1. **ScreenCaptureKit requires macOS 13+** (Ventura). iPhone Mirroring requires macOS 15+ (Sequoia). Target macOS 15.
2. **Screen Recording permission** must be granted by the user. The app should gracefully prompt for this on first launch.
3. **iPhone Mirroring window name** in SCShareableContent may appear as "iPhone" or the device name — discover it dynamically, don't hardcode.
4. **DRM content:** Some streaming audio (Spotify, Apple Music) may be excluded from capture. TikTok audio is NOT DRM-protected and captures fine.
5. **Whisper turbo cannot translate directly** — it only transcribes. Always transcribe first, then pass text to NLLB.
6. **NLLB input limit:** 512 tokens max. For a 2-second TikTok clip this is never an issue, but truncate just in case.
7. **Python service startup time:** Loading both models takes ~10–15 seconds on first launch. Show a "warming up..." state in the menu bar icon.
8. **Chunk overlap:** Implement a 0.5s rolling overlap between audio chunks to avoid cutting mid-word at boundaries.

---

## Launch Sequence

1. User launches the macOS app
2. App starts Python microservice as a subprocess (`subprocess.Popen`)
3. App polls `localhost:5005/health` until service is ready
4. Menu bar icon changes from grey → active
5. User opens iPhone Mirroring and plays a TikTok video
6. App detects iPhone Mirroring window via ScreenCaptureKit
7. Audio capture starts automatically
8. Subtitles appear in the floating overlay window

---

## First Steps for Claude Code

1. Scaffold the Xcode project as a macOS menu bar app (`LSUIElement = YES` in Info.plist)
2. Add Screen Recording entitlement
3. Implement `AudioCaptureManager.swift` using `SCStream` targeting the iPhone Mirroring app
4. Build the Python FastAPI service with lazy model loading
5. Implement the `SubtitleOverlayWindow` as a frameless `NSPanel`
6. Wire everything together with the `TranslationClient`
7. Test with a Georgian TikTok video via iPhone Mirroring

---

## Out of Scope (for now)

- Other Kartvelian languages (Mingrelian, Svan, Laz) — no adequate free models exist yet
- TTS readback of translated audio — display subtitles only
- App Store distribution — build for local personal use first
- Multi-language simultaneous translation — single source language per session
