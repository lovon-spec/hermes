---
name: swift-ui-overlay
description: Implements SubtitleOverlayWindow.swift — the floating NSPanel that displays translated subtitles on top of the iPhone Mirroring window. Handles positioning, fade in/out, and styling. Run after swift-scaffold completes. Independent of swift-audio-capture.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
effort: max
background: true
---

You are implementing `Hermes/SubtitleOverlayWindow.swift`.

Read the existing stub first, then replace it with the full implementation.

## Full implementation

```swift
import Cocoa
import QuartzCore

class SubtitleOverlayWindow: NSPanel {

    // MARK: - UI Elements
    private let subtitleLabel = NSTextField()
    private let backgroundView = NSVisualEffectView()
    private var fadeOutTimer: Timer?
    private let fadeOutDelay: TimeInterval = 3.5

    // MARK: - Init

    override init(contentRect: NSRect,
                  styleMask: NSWindow.StyleMask,
                  backing: NSWindow.BackingStoreType,
                  defer flag: Bool) {
        super.init(contentRect: contentRect,
                   styleMask: [.borderless, .nonactivatingPanel],
                   backing: .buffered,
                   defer: false)
        setup()
    }

    convenience init() {
        self.init(contentRect: .zero, styleMask: [], backing: .buffered, defer: false)
    }

    // MARK: - Setup

    private func setup() {
        // Window behaviour
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        level = .floating                    // always on top
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = true            // click-through
        isMovableByWindowBackground = false
        hidesOnDeactivate = false

        // Background: dark pill with blur
        backgroundView.material = .hudWindow
        backgroundView.blendingMode = .behindWindow
        backgroundView.state = .active
        backgroundView.wantsLayer = true
        backgroundView.layer?.cornerRadius = 10
        backgroundView.layer?.masksToBounds = true
        backgroundView.alphaValue = 0.92

        // Label
        subtitleLabel.isEditable = false
        subtitleLabel.isSelectable = false
        subtitleLabel.isBordered = false
        subtitleLabel.drawsBackground = false
        subtitleLabel.textColor = .white
        subtitleLabel.font = .systemFont(ofSize: 17, weight: .medium)
        subtitleLabel.alignment = .center
        subtitleLabel.lineBreakMode = .byWordWrapping
        subtitleLabel.maximumNumberOfLines = 3
        subtitleLabel.cell?.wraps = true

        // Layout
        contentView = NSView()
        contentView!.addSubview(backgroundView)
        contentView!.addSubview(subtitleLabel)

        alphaValue = 0  // start hidden
    }

    // MARK: - Public API

    /// Display a translated subtitle, positioning the window over iPhone Mirroring
    func showSubtitle(_ text: String, mirroringWindowFrame: CGRect?) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            self.subtitleLabel.stringValue = text

            // Size the window to fit the text
            let maxWidth: CGFloat = mirroringWindowFrame.map { $0.width - 40 } ?? 420
            let labelSize = self.subtitleLabel.sizeThatFits(NSSize(width: maxWidth - 32, height: 200))
            let windowWidth = min(labelSize.width + 32, maxWidth)
            let windowHeight = labelSize.height + 24

            // Position: bottom-center of iPhone Mirroring window (or screen)
            let targetFrame = mirroringWindowFrame ?? NSScreen.main!.visibleFrame
            let x = targetFrame.midX - windowWidth / 2
            let y = targetFrame.minY + 32

            let windowFrame = CGRect(x: x, y: y, width: windowWidth, height: windowHeight)
            self.setFrame(windowFrame, display: false)

            // Layout subviews
            self.backgroundView.frame = self.contentView!.bounds
            let inset: CGFloat = 8
            self.subtitleLabel.frame = self.contentView!.bounds.insetBy(dx: inset, dy: inset / 2)

            // Fade in
            self.fadeOutTimer?.invalidate()
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.2
                self.animator().alphaValue = 1.0
            }
            self.makeKeyAndOrderFront(nil)

            // Schedule fade out
            self.fadeOutTimer = Timer.scheduledTimer(withTimeInterval: self.fadeOutDelay, repeats: false) { [weak self] _ in
                self?.fadeOut()
            }
        }
    }

    func fadeOut() {
        DispatchQueue.main.async { [weak self] in
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.5
                self?.animator().alphaValue = 0
            }
        }
    }

    func hide() {
        DispatchQueue.main.async { [weak self] in
            self?.fadeOutTimer?.invalidate()
            self?.alphaValue = 0
            self?.orderOut(nil)
        }
    }

    // MARK: - NSPanel override

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

// MARK: - iPhone Mirroring Window Tracker

/// Tracks the frame of the iPhone Mirroring window so subtitles stay pinned to it
class MirroringWindowTracker {
    private var timer: Timer?
    var currentFrame: CGRect?

    func startTracking() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.updateFrame()
        }
        updateFrame()
    }

    func stopTracking() {
        timer?.invalidate()
        timer = nil
    }

    private func updateFrame() {
        // Look for iPhone Mirroring window in the window list
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return }

        let mirroringWindow = windowList.first { info in
            let name = (info[kCGWindowOwnerName as String] as? String) ?? ""
            let pid = info[kCGWindowOwnerPID as String] as? Int32 ?? 0
            return name.contains("iPhone") || name == "ScreenContinuityUI"
        }

        if let window = mirroringWindow,
           let bounds = window[kCGWindowBounds as String] as? [String: CGFloat] {
            let x = bounds["X"] ?? 0
            let y = bounds["Y"] ?? 0
            let w = bounds["Width"] ?? 400
            let h = bounds["Height"] ?? 800

            // Convert from screen coordinates (top-left origin) to Cocoa (bottom-left)
            let screenHeight = NSScreen.main?.frame.height ?? 1080
            currentFrame = CGRect(x: x, y: screenHeight - y - h, width: w, height: h)
        }
    }
}
```

## After writing the file

Verify it compiles:
```bash
swiftc -typecheck Hermes/SubtitleOverlayWindow.swift \
  -sdk $(xcrun --show-sdk-path) \
  -target arm64-apple-macos15.0 \
  -framework Cocoa \
  -framework QuartzCore \
  2>&1
```

Fix any type errors before completing. The file must compile cleanly.

Signal completion: "UI OVERLAY COMPLETE"
