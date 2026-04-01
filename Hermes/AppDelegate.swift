import Cocoa
import ScreenCaptureKit

class AppDelegate: NSObject, NSApplicationDelegate {

    // MARK: - Components

    private var statusItem: NSStatusItem?
    private var audioManager: AudioCaptureManager?
    private var overlayWindow: SubtitleOverlayWindow?
    private var translationClient: TranslationClient?
    private var windowTracker: MirroringWindowTracker?
    private var pythonProcess: Process?

    // MARK: - Menu items (need references for enable/disable)

    private var startMenuItem: NSMenuItem?
    private var stopMenuItem: NSMenuItem?
    private var statusMenuItem: NSMenuItem?

    // MARK: - State

    private var isCapturing = false
    private var isServiceReady = false

    /// Timer that syncs MirroringWindowTracker.currentFrame to the overlay
    private var overlayTrackingTimer: Timer?

    // MARK: - App Lifecycle

    private let logFile = "/tmp/hermes-debug.log"

    private func log(_ message: String) {
        let line = "\(Date()): \(message)\n"
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logFile) {
                if let handle = FileHandle(forWritingAtPath: logFile) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    handle.closeFile()
                }
            } else {
                FileManager.default.createFile(atPath: logFile, contents: data)
            }
        }
        print(message)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        log("applicationDidFinishLaunching called")
        setupMenuBar()
        log("menu bar set up")
        setupOverlayWindow()
        log("overlay window set up")
        startPythonService()
        log("startPythonService called")
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopEverything()
    }

    // MARK: - Menu Bar

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        guard let button = statusItem?.button else { return }

        button.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Hermes Translator")
        button.image?.isTemplate = true

        let menu = NSMenu()

        // Status line (non-interactive)
        let status = NSMenuItem(title: "Warming up AI models...", action: nil, keyEquivalent: "")
        status.isEnabled = false
        menu.addItem(status)
        statusMenuItem = status

        menu.addItem(NSMenuItem.separator())

        // Start
        let start = NSMenuItem(title: "Start Translation", action: #selector(startTranslation), keyEquivalent: "s")
        start.target = self
        start.isEnabled = false  // disabled until service is ready
        menu.addItem(start)
        startMenuItem = start

        // Stop
        let stop = NSMenuItem(title: "Stop Translation", action: #selector(stopTranslation), keyEquivalent: "x")
        stop.target = self
        stop.isEnabled = false
        stop.isHidden = true
        menu.addItem(stop)
        stopMenuItem = stop

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quit = NSMenuItem(title: "Quit Hermes", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)

        statusItem?.menu = menu
    }

    private func updateMenuForState() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            if self.isCapturing {
                self.statusMenuItem?.title = "Translating..."
                self.startMenuItem?.isHidden = true
                self.startMenuItem?.isEnabled = false
                self.stopMenuItem?.isHidden = false
                self.stopMenuItem?.isEnabled = true
            } else if self.isServiceReady {
                self.statusMenuItem?.title = "Ready"
                self.startMenuItem?.isHidden = false
                self.startMenuItem?.isEnabled = true
                self.stopMenuItem?.isHidden = true
                self.stopMenuItem?.isEnabled = false
            } else {
                self.statusMenuItem?.title = "Warming up AI models..."
                self.startMenuItem?.isEnabled = false
                self.stopMenuItem?.isEnabled = false
                self.startMenuItem?.isHidden = false
                self.stopMenuItem?.isHidden = true
            }
        }
    }

    // MARK: - Python Service Lifecycle

    private func startPythonService() {
        // Locate translator_service/main.py relative to the project
        guard let serviceDir = findTranslatorServiceDirectory() else {
            showServiceStatus("translator_service/ not found")
            print("[AppDelegate] Could not find translator_service/ directory")
            return
        }

        // Find a working python3 binary
        let pythonPaths = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]
        guard let pythonPath = pythonPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            showServiceStatus("python3 not found")
            print("[AppDelegate] No python3 found in standard paths")
            return
        }

        print("[AppDelegate] Using python3 at: \(pythonPath)")
        print("[AppDelegate] Service directory: \(serviceDir)")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["main.py"]
        process.currentDirectoryURL = URL(fileURLWithPath: serviceDir)

        // Redirect stdout/stderr so we can see model loading progress
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Log stderr asynchronously for debugging
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                print("[Python stderr] \(text)", terminator: "")
            }
        }
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                print("[Python stdout] \(text)", terminator: "")
            }
        }

        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                print("[AppDelegate] Python process exited with code \(proc.terminationStatus)")
                self?.isServiceReady = false
                self?.showServiceStatus("Service stopped (exit \(proc.terminationStatus))")
                self?.updateMenuForState()
            }
        }

        do {
            try process.run()
            pythonProcess = process
            print("[AppDelegate] Python service process started (pid \(process.processIdentifier))")
        } catch {
            showServiceStatus("Failed to start: \(error.localizedDescription)")
            print("[AppDelegate] Failed to launch Python process: \(error)")
            return
        }

        // Start polling for readiness
        translationClient = TranslationClient()
        translationClient?.pollUntilReady(interval: 2.0) { [weak self] in
            guard let self = self else { return }
            print("[AppDelegate] Translation service is ready")
            self.isServiceReady = true
            self.updateMenuForState()
        }
    }

    /// Walk up from the executable to find translator_service/ in the project tree.
    /// Works for both Xcode debug builds (.app/Contents/MacOS/Hermes) and direct execution.
    private func findTranslatorServiceDirectory() -> String? {
        let fm = FileManager.default

        // Candidate directories to check
        var candidates: [String] = []

        // 1. Relative to the executable (Xcode build: .../Build/Products/Debug/Hermes.app/Contents/MacOS/Hermes)
        let executableURL = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0]).resolvingSymlinksInPath()
        var dir = executableURL.deletingLastPathComponent()  // MacOS/
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("translator_service").path
            candidates.append(candidate)
            dir = dir.deletingLastPathComponent()
        }

        // 2. Relative to Bundle.main.bundlePath
        let bundleURL = URL(fileURLWithPath: Bundle.main.bundlePath)
        dir = bundleURL
        for _ in 0..<8 {
            let candidate = dir.appendingPathComponent("translator_service").path
            candidates.append(candidate)
            dir = dir.deletingLastPathComponent()
        }

        for candidate in candidates {
            let mainPy = (candidate as NSString).appendingPathComponent("main.py")
            if fm.fileExists(atPath: mainPy) {
                return candidate
            }
        }

        return nil
    }

    private func showServiceStatus(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.statusMenuItem?.title = message
        }
    }

    // MARK: - Start / Stop Translation

    @objc private func startTranslation() {
        guard isServiceReady, !isCapturing else { return }
        log("startTranslation clicked")

        Task {
            do {
                let manager = AudioCaptureManager()

                // Wire audio manager's translation callback to the subtitle overlay
                manager.onTranslation = { [weak self] text in
                    Task { @MainActor in
                        self?.overlayWindow?.showSubtitle(text)
                    }
                }

                try await manager.startCapture()
                log("capture started successfully")

                await MainActor.run {
                    self.audioManager = manager
                    self.isCapturing = true
                    self.updateMenuForState()
                    self.windowTracker?.startTracking()
                    self.startOverlayTracking()
                    self.log("tracking started")
                }
            } catch CaptureError.iphoneMirroringNotFound {
                log("ERROR: iPhone Mirroring not found")
                await MainActor.run {
                    self.showAlert(
                        title: "iPhone Mirroring Not Found",
                        message: "Please open iPhone Mirroring first, then try again."
                    )
                }
            } catch {
                log("ERROR: capture failed: \(error)")
                await MainActor.run {
                    self.showAlert(
                        title: "Capture Error",
                        message: error.localizedDescription
                    )
                }
            }
        }
    }

    @objc private func stopTranslation() {
        guard isCapturing else { return }

        Task {
            await audioManager?.stopCapture()
            await MainActor.run {
                self.audioManager = nil
                self.isCapturing = false
                self.updateMenuForState()
                self.windowTracker?.stopTracking()
                self.stopOverlayTracking()
                self.overlayWindow?.hideSubtitle()
            }
        }
    }

    // MARK: - Overlay Window Setup

    private func setupOverlayWindow() {
        overlayWindow = SubtitleOverlayWindow()
        windowTracker = MirroringWindowTracker()
    }

    /// Periodically sync the window tracker frame to the overlay position
    private func startOverlayTracking() {
        overlayTrackingTimer?.invalidate()
        overlayTrackingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self,
                  let frame = self.windowTracker?.currentFrame else { return }
            self.overlayWindow?.trackWindow(frame: frame)
        }
    }

    private func stopOverlayTracking() {
        overlayTrackingTimer?.invalidate()
        overlayTrackingTimer = nil
    }

    // MARK: - Cleanup

    private func stopEverything() {
        // Stop capture
        if isCapturing {
            Task {
                await audioManager?.stopCapture()
            }
        }
        audioManager = nil
        isCapturing = false

        // Stop tracking
        windowTracker?.stopTracking()
        stopOverlayTracking()

        // Hide overlay
        overlayWindow?.hideSubtitle()

        // Stop health polling
        translationClient?.stopPolling()
        translationClient = nil

        // Terminate Python process
        if let process = pythonProcess, process.isRunning {
            print("[AppDelegate] Terminating Python service (pid \(process.processIdentifier))")
            process.terminate()
        }
        pythonProcess = nil
    }

    // MARK: - Alerts

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
