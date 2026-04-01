# Hermes — Orchestration Guide

This project builds a macOS menu bar app that captures iPhone Mirroring audio via
ScreenCaptureKit and displays real-time translated subtitles in a floating overlay window.
Works across every app on your iPhone — TikTok, YouTube, Instagram, WeChat, any streaming
or social app — for any language those apps don't natively support.

Everything runs **100% locally and for free** — no API keys, no internet required after setup.

Read `CLAUDE_CODE_PRIMER.md` for full architecture details before starting.

---

## Agent Execution Plan

Build in three phases. Phase 1 agents run in **parallel**. Do not wait for one to finish
before starting the other in the same phase.

### Phase 1 — spawn both simultaneously (background)

```
@python-microservice   Build the entire Python FastAPI service
@swift-scaffold        Create the Xcode project skeleton
```

These are completely independent. Spawn both as background tasks immediately.

### Phase 2 — after swift-scaffold completes, spawn both simultaneously (background)

```
@swift-audio-capture   Implement ScreenCaptureKit audio capture
@swift-ui-overlay      Implement NSPanel floating subtitle window
```

These both depend on the Xcode project existing but are independent of each other.
Spawn both together as soon as swift-scaffold finishes.

### Phase 3 — after ALL phase 2 agents complete

```
@swift-integration     Wire everything together, implement AppDelegate,
                       add Python subprocess launcher, end-to-end test
```

---

## Final Verification Checklist

After swift-integration completes, verify:

- [ ] `translator_service/` exists with `main.py`, `whisper_engine.py`, `nllb_engine.py`, `requirements.txt`
- [ ] Xcode project builds without errors (macOS 15+ target)
- [ ] Screen Recording entitlement is present in `.entitlements` file
- [ ] `LSUIElement = YES` in Info.plist (pure menu bar app, no Dock icon)
- [ ] App finds iPhone Mirroring window dynamically (not hardcoded name)
- [ ] Python service starts on `localhost:5005` and `/health` returns 200
- [ ] Subtitle overlay appears at bottom of iPhone Mirroring window
- [ ] End-to-end: audio in any language → English subtitle appears within ~1 second

---

## Key Files Reference

```
Hermes/
├── CLAUDE.md                          ← you are here
├── CLAUDE_CODE_PRIMER.md             ← full architecture reference
├── .claude/agents/                   ← subagent definitions
│   ├── python-microservice.md
│   ├── swift-scaffold.md
│   ├── swift-audio-capture.md
│   ├── swift-ui-overlay.md
│   └── swift-integration.md
├── Hermes.xcodeproj
├── Hermes/
│   ├── AppDelegate.swift
│   ├── AudioCaptureManager.swift
│   ├── TranslationClient.swift
│   ├── SubtitleOverlayWindow.swift
│   └── Info.plist
└── translator_service/
    ├── main.py
    ├── whisper_engine.py
    ├── nllb_engine.py
    └── requirements.txt
```
