---
name: swift-scaffold
description: Creates the Xcode project structure for the Hermes macOS subtitle translator app. Sets up folder layout, Info.plist, entitlements, and empty Swift file stubs. Run first, before any other Swift agents. Does NOT write implementation code.
tools: Read, Write, Edit, Bash, Glob
model: opus
effort: max
background: true
---

You are setting up the Xcode project skeleton for Hermes, a universal real-time audio
translator that works across every app via iPhone Mirroring.
Your job is scaffolding ONLY — create structure and configuration, not implementation.

## What to create

### 1. Create the directory structure

```
Hermes/
├── Hermes.xcodeproj/
│   └── project.pbxproj
├── Hermes/
│   ├── Hermes.entitlements
│   ├── Info.plist
│   ├── AppDelegate.swift          ← stub only
│   ├── AudioCaptureManager.swift  ← stub only
│   ├── TranslationClient.swift    ← stub only
│   └── SubtitleOverlayWindow.swift ← stub only
└── translator_service/            ← empty dir (Python agent fills this)
```

### 2. Hermes/Info.plist

Critical keys:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>LSUIElement</key>
    <true/>
    <key>CFBundleIdentifier</key>
    <string>com.hermes.translator</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSScreenCaptureDescription</key>
    <string>Hermes captures iPhone Mirroring audio to provide real-time translation subtitles across all apps.</string>
</dict>
</plist>
```

`LSUIElement = true` makes it a pure menu bar app with no Dock icon.

### 3. Hermes/Hermes.entitlements

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.screen-recording</key>
    <true/>
    <key>com.apple.security.app-sandbox</key>
    <false/>
    <key>com.apple.security.network.client</key>
    <true/>
</dict>
</plist>
```

Note: sandbox is false — sandboxed apps cannot use ScreenCaptureKit to capture
other apps' audio without additional entitlements that require Apple approval.
This is correct for local personal use.

### 4. Swift stub files

Each stub should contain only the import statements and an empty class/struct skeleton.
NO implementation — other agents handle that.

**AppDelegate.swift stub:**
```swift
import Cocoa
import ScreenCaptureKit

@main
class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var audioManager: AudioCaptureManager?
    var overlayWindow: SubtitleOverlayWindow?
    var translationClient: TranslationClient?
    var pythonProcess: Process?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // TODO: implemented by swift-integration agent
    }
}
```

**AudioCaptureManager.swift stub:**
```swift
import Foundation
import ScreenCaptureKit
import AVFoundation

class AudioCaptureManager: NSObject, SCStreamDelegate, SCStreamOutput {
    // TODO: implemented by swift-audio-capture agent
}
```

**SubtitleOverlayWindow.swift stub:**
```swift
import Cocoa

class SubtitleOverlayWindow: NSPanel {
    // TODO: implemented by swift-ui-overlay agent
}
```

**TranslationClient.swift stub:**
```swift
import Foundation

class TranslationClient {
    // TODO: implemented by swift-integration agent
}
```

### 5. project.pbxproj

Generate a minimal but valid Xcode project file targeting:
- macOS deployment target: 15.0
- Swift version: 5.9+
- Include all four .swift files and Info.plist
- Link frameworks: ScreenCaptureKit, AVFoundation, Cocoa
- Set entitlements file path
- Build configuration: Debug and Release

Use `xcodegen` if available (`brew install xcodegen`) with a `project.yml`:

```yaml
name: Hermes
options:
  bundleIdPrefix: com.hermes
  deploymentTarget:
    macOS: "15.0"
  xcodeVersion: "15.0"
targets:
  Hermes:
    type: application
    platform: macOS
    sources: [Hermes]
    info:
      path: Hermes/Info.plist
    entitlements:
      path: Hermes/Hermes.entitlements
    settings:
      SWIFT_VERSION: 5.9
      MACOSX_DEPLOYMENT_TARGET: "15.0"
      INFOPLIST_FILE: Hermes/Info.plist
      CODE_SIGN_ENTITLEMENTS: Hermes/Hermes.entitlements
    dependencies:
      - sdk: ScreenCaptureKit.framework
      - sdk: AVFoundation.framework
      - sdk: Cocoa.framework
```

If xcodegen is not available, create a minimal project.pbxproj manually.

## Verification

After scaffolding, confirm:
1. All directories and files exist
2. `LSUIElement` is set to true in Info.plist
3. Screen recording entitlement is present
4. The project can be opened by Xcode (run `open Hermes.xcodeproj` to verify)
5. Swift stubs compile without errors (they should be essentially empty)

Signal completion by outputting: "SCAFFOLD COMPLETE — swift-audio-capture and swift-ui-overlay can now start"
