import Cocoa
import QuartzCore

class SubtitleOverlayWindow: NSPanel {

    // MARK: - UI Elements
    private let subtitleLabel = NSTextField()
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

        // Drop shadow on text for readability
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.8)
        shadow.shadowOffset = NSSize(width: 0, height: -1)
        shadow.shadowBlurRadius = 3
        subtitleLabel.shadow = shadow

        // Layout
        contentView = NSView()
        contentView!.wantsLayer = true
        contentView!.addSubview(backgroundView)
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

    /// Display a translated subtitle, positioning the window over iPhone Mirroring
    func showSubtitle(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.subtitleLabel.stringValue = text
            self.repositionWindow()

            // Fade in
            self.fadeOutTimer?.invalidate()
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.2
                self.animator().alphaValue = 1.0
            }
            self.orderFrontRegardless()

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

        // Size the window to fit the text
        let maxWidth: CGFloat = targetFrame.width - 40
        let horizontalPadding: CGFloat = 24  // 12pt each side
        let verticalPadding: CGFloat = 16    // 8pt each side
        let labelSize = subtitleLabel.sizeThatFits(NSSize(width: maxWidth - horizontalPadding, height: 200))
        let windowWidth = min(labelSize.width + horizontalPadding, maxWidth)
        let windowHeight = labelSize.height + verticalPadding

        // Position: bottom-center of target window
        let x = targetFrame.midX - windowWidth / 2
        let y = targetFrame.minY + 32

        let windowFrame = CGRect(x: x, y: y, width: windowWidth, height: windowHeight)
        setFrame(windowFrame, display: false)

        // Layout subviews
        backgroundView.frame = contentView!.bounds
        subtitleLabel.frame = contentView!.bounds.insetBy(dx: 12, dy: 8)
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
