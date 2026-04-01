import Cocoa
import QuartzCore

class SubtitleOverlayWindow: NSPanel {

    // MARK: - UI Elements
    private let subtitleLabel = NSTextField()
    private let originalLabel = NSTextField()
    private let backgroundView = NSVisualEffectView()
    private var fadeOutTimer: Timer?
    private let fadeOutDelay: TimeInterval = 3.5
    private var trackedFrame: CGRect?

    // MARK: - Public State

    /// Whether the subtitle overlay is currently visible
    var isShowingSubtitle: Bool {
        return alphaValue > 0 && isVisible
    }

    // MARK: - Init

    override init(contentRect: NSRect,
                  styleMask style: NSWindow.StyleMask,
                  backing backingStoreType: NSWindow.BackingStoreType,
                  defer flag: Bool) {
        super.init(contentRect: contentRect,
                   styleMask: [.borderless, .nonactivatingPanel, .hudWindow],
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
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        ignoresMouseEvents = true
        isMovableByWindowBackground = false
        hidesOnDeactivate = false
        sharingType = .none  // Do not appear in screen recordings or screenshots

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
        subtitleLabel.font = .systemFont(ofSize: 18, weight: .bold)
        subtitleLabel.alignment = .center
        subtitleLabel.lineBreakMode = .byWordWrapping
        subtitleLabel.maximumNumberOfLines = 3
        subtitleLabel.cell?.wraps = true

        // Original text label (smaller, dimmer, shown above translation for foreign languages)
        originalLabel.isEditable = false
        originalLabel.isSelectable = false
        originalLabel.isBordered = false
        originalLabel.drawsBackground = false
        originalLabel.textColor = NSColor.white.withAlphaComponent(0.6)
        originalLabel.font = .systemFont(ofSize: 14, weight: .medium)
        originalLabel.alignment = .center
        originalLabel.lineBreakMode = .byWordWrapping
        originalLabel.maximumNumberOfLines = 2
        originalLabel.cell?.wraps = true
        originalLabel.isHidden = true

        // Drop shadow on text for readability
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.8)
        shadow.shadowOffset = NSSize(width: 0, height: -1)
        shadow.shadowBlurRadius = 3
        subtitleLabel.shadow = shadow
        originalLabel.shadow = shadow

        // Layout
        contentView = NSView()
        contentView!.wantsLayer = true
        contentView!.addSubview(backgroundView)
        contentView!.addSubview(originalLabel)
        contentView!.addSubview(subtitleLabel)

        alphaValue = 0  // start hidden
    }

    // MARK: - Public API

    /// Reposition the overlay to track the iPhone Mirroring window
    func trackWindow(frame: CGRect) {
        trackedFrame = frame
        // If currently showing, reposition immediately
        if isShowingSubtitle {
            repositionWindow()
        }
    }

    private func overlayLog(_ message: String) {
        let line = "\(Date()): [Overlay] \(message)\n"
        if let data = line.data(using: .utf8),
           let handle = FileHandle(forWritingAtPath: "/tmp/hermes-debug.log") {
            handle.seekToEndOfFile()
            handle.write(data)
            handle.closeFile()
        }
    }

    /// Display a translated subtitle, with optional original transcription for foreign languages
    func showSubtitle(_ text: String, original: String? = nil) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.subtitleLabel.stringValue = text
            if let orig = original, !orig.isEmpty, orig != text {
                self.originalLabel.stringValue = orig
                self.originalLabel.isHidden = false
            } else {
                self.originalLabel.isHidden = true
            }
            self.repositionWindow()

            // Fade in
            self.fadeOutTimer?.invalidate()
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.2
                self.animator().alphaValue = 1.0
            }
            self.orderFrontRegardless()
            self.overlayLog("frame=\(self.frame) alpha=\(self.alphaValue) visible=\(self.isVisible) text=\(text.prefix(40))")

            // Schedule fade out
            self.fadeOutTimer = Timer.scheduledTimer(withTimeInterval: self.fadeOutDelay, repeats: false) { [weak self] _ in
                self?.fadeOut()
            }
        }
    }

    /// Immediately hide the subtitle overlay
    func hideSubtitle() {
        DispatchQueue.main.async { [weak self] in
            self?.fadeOutTimer?.invalidate()
            self?.alphaValue = 0
            self?.orderOut(nil)
        }
    }

    // MARK: - Private

    private func repositionWindow() {
        let targetFrame = trackedFrame ?? NSScreen.main?.visibleFrame ?? CGRect(x: 0, y: 0, width: 600, height: 400)

        let horizontalPadding: CGFloat = 24
        let verticalPadding: CGFloat = 16
        let maxWidth: CGFloat = max(targetFrame.width - 40, 200)
        let textMaxWidth = maxWidth - horizontalPadding

        // Measure translation label
        let translationSize: NSSize
        if let cell = subtitleLabel.cell {
            translationSize = cell.cellSize(forBounds: NSRect(x: 0, y: 0, width: textMaxWidth, height: 200))
        } else {
            translationSize = NSSize(width: textMaxWidth, height: 22)
        }

        // Measure original label if visible
        var originalSize = NSSize.zero
        let showingOriginal = !originalLabel.isHidden
        if showingOriginal, let cell = originalLabel.cell {
            originalSize = cell.cellSize(forBounds: NSRect(x: 0, y: 0, width: textMaxWidth, height: 100))
        }

        let spacing: CGFloat = showingOriginal ? 4 : 0
        let contentHeight = translationSize.height + (showingOriginal ? originalSize.height + spacing : 0)
        let windowWidth = min(max(translationSize.width, originalSize.width) + horizontalPadding, maxWidth)
        let windowHeight = max(contentHeight + verticalPadding, 40)

        // Position: bottom-center of target window
        let x = targetFrame.midX - windowWidth / 2
        let y = targetFrame.minY + 32

        let windowFrame = CGRect(x: x, y: y, width: windowWidth, height: windowHeight)
        setFrame(windowFrame, display: true)

        // Layout subviews
        backgroundView.frame = contentView!.bounds
        let inset = contentView!.bounds.insetBy(dx: 12, dy: 8)
        if showingOriginal {
            originalLabel.frame = NSRect(x: inset.minX, y: inset.minY + translationSize.height + spacing,
                                         width: inset.width, height: originalSize.height)
            subtitleLabel.frame = NSRect(x: inset.minX, y: inset.minY,
                                         width: inset.width, height: translationSize.height)
        } else {
            subtitleLabel.frame = inset
        }
    }

    private func fadeOut() {
        DispatchQueue.main.async { [weak self] in
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.5
                self?.animator().alphaValue = 0
            }
        }
    }

    // MARK: - NSPanel overrides

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
