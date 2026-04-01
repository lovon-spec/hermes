---
name: swift-integration
description: Wires all components together. Implements AppDelegate.swift and TranslationClient.swift. Launches the Python microservice as a subprocess, connects audio capture to translation, and displays results in the overlay. Run ONLY after swift-audio-capture, swift-ui-overlay, and python-microservice have all completed.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
effort: max
---

You are the final integration agent. Read ALL existing files first before writing anything.

Read these files in order:
1. `CLAUDE_CODE_PRIMER.md` (architecture reference)
2. `Hermes/AudioCaptureManager.swift` (understand the API)
3. `Hermes/SubtitleOverlayWindow.swift` (understand the API)
4. `translator_service/main.py` (understand the HTTP endpoint)

Then implement the two remaining files.

---

## 1. Hermes/TranslationClient.swift

HTTP client that sends PCM chunks to the Python service and returns translated text.

```swift
import Foundation

struct TranslationResponse: Decodable {
    let translation: String
    let sourceLang: String
    let originalText: String
    let latencyMs: Int
    let skipped: Bool

    enum CodingKeys: String, CodingKey {
        case translation
        case sourceLang = "source_lang"
        case originalText = "original_text"
        case latencyMs = "latency_ms"
        case skipped
    }
}

class TranslationClient {
    private let baseURL = URL(string: "http://127.0.0.1:5005")!
    private let session: URLSession
    private var isServiceReady = false

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 15
        session = URLSession(configuration: config)
    }

    // MARK: - Health check

    /// Poll /health until the Python service signals readiness
    func waitUntilReady(timeout: TimeInterval = 60, completion: @escaping (Bool) -> Void) {
        let deadline = Date().addingTimeInterval(timeout)
        pollHealth(until: deadline, completion: completion)
    }

    private func pollHealth(until deadline: Date, completion: @escaping (Bool) -> Void) {
        guard Date() < deadline else { completion(false); return }

        let url = baseURL.appendingPathComponent("health")
        URLSession.shared.dataTask(with: url) { [weak self] data, _, error in
            if let data = data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
               json["status"] == "ready" {
                self?.isServiceReady = true
                completion(true)
            } else {
                DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
                    self?.pollHealth(until: deadline, completion: completion)
                }
            }
        }.resume()
    }

    // MARK: - Translation

    /// Send raw PCM bytes to /translate and return the translation result
    func translate(pcmData: Data, completion: @escaping (Result<TranslationResponse, Error>) -> Void) {
        guard isServiceReady else { return }

        var request = URLRequest(url: baseURL.appendingPathComponent("translate"))
        request.httpMethod = "POST"
        request.setValue("audio/raw", forHTTPHeaderField: "Content-Type")
        request.httpBody = pcmData

        session.dataTask(with: request) { data, response, error in
            if let error = error { completion(.failure(error)); return }
            guard let data = data else {
                completion(.failure(NSError(domain: "TranslationClient", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Empty response"])))
                return
            }
            do {
                let result = try JSONDecoder().decode(TranslationResponse.self, from: data)
                completion(.success(result))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}
```

---

## 2. Hermes/AppDelegate.swift

Full implementation — menu bar, Python subprocess lifecycle, wiring audio → translation → UI.

```swift
import Cocoa
import ScreenCaptureKit

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    // MARK: - Components
    private var statusItem: NSStatusItem?
    private var audioManager: AudioCaptureManager?
    private var overlayWindow: SubtitleOverlayWindow?
    private var translationClient: TranslationClient?
    private var windowTracker: MirroringWindowTracker?
    private var pythonProcess: Process?

    // MARK: - State
    private var isCapturing = false
    private var isServiceReady = false

    // MARK: - App Launch

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        setupOverlayWindow()
        startPythonService()
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopEverything()
    }

    // MARK: - Menu Bar

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        guard let button = statusItem?.button else { return }

        button.image = NSImage(systemSymbolName: "captions.bubble", accessibilityDescription: "Translator")
        button.image?.isTemplate = true

        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Starting service...", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        statusItem?.menu = menu
    }

    private func updateMenuBar(state: MenuState) {
        DispatchQueue.main.async { [weak self] in
            guard let self, let menu = self.statusItem?.menu else { return }

            let title: String
            let action: Selector?

            switch state {
            case .loading:
                title = "Starting AI models (~15s)..."
                action = nil
            case .ready:
                title = "Start Translating"
                action = #selector(self.toggleCapture)
            case .capturing:
                title = "Stop Translating"
                action = #selector(self.toggleCapture)
            case .error(let msg):
                title = "Error: \(msg)"
                action = nil
            }

            menu.items[0].title = title
            menu.items[0].action = action
            menu.items[0].target = action != nil ? self : nil
        }
    }

    @objc private func toggleCapture() {
        isCapturing ? stopCapture() : startCapture()
    }

    enum MenuState {
        case loading, ready, capturing, error(String)
    }

    // MARK: - Python Service

    private func startPythonService() {
        updateMenuBar(state: .loading)

        // Find python3 in common locations
        let pythonPaths = ["/usr/bin/python3", "/opt/homebrew/bin/python3",
                           "/usr/local/bin/python3", "/opt/anaconda3/bin/python3"]
        guard let python = pythonPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            updateMenuBar(state: .error("python3 not found"))
            return
        }

        let serviceDir = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("translator_service")
            .path

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                             "--port", "5005", "--log-level", "warning"]
        process.currentDirectoryURL = URL(fileURLWithPath: serviceDir)
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.updateMenuBar(state: .error("Python service stopped"))
            }
        }

        do {
            try process.run()
            pythonProcess = process
        } catch {
            updateMenuBar(state: .error("Failed to start service: \(error.localizedDescription)"))
            return
        }

        // Wait for the service to be ready
        translationClient = TranslationClient()
        translationClient?.waitUntilReady(timeout: 90) { [weak self] ready in
            DispatchQueue.main.async {
                if ready {
                    self?.isServiceReady = true
                    self?.updateMenuBar(state: .ready)
                } else {
                    self?.updateMenuBar(state: .error("Service failed to start in time"))
                }
            }
        }
    }

    private func stopEverything() {
        stopCapture()
        pythonProcess?.terminate()
        pythonProcess = nil
    }

    // MARK: - Capture Control

    private func startCapture() {
        guard isServiceReady else { return }

        // Request screen recording permission if needed
        Task {
            do {
                try await SCShareableContent.current  // triggers permission prompt if needed
                let manager = AudioCaptureManager()
                manager.onChunkReady = { [weak self] pcmData in
                    self?.handleAudioChunk(pcmData)
                }
                try await manager.startCapture()
                await MainActor.run {
                    self.audioManager = manager
                    self.isCapturing = true
                    self.updateMenuBar(state: .capturing)
                    self.windowTracker?.startTracking()
                }
            } catch CaptureError.iphoneMirroringNotFound {
                await MainActor.run {
                    self.showError("iPhone Mirroring is not running.\nOpen iPhone Mirroring and try again.")
                }
            } catch {
                await MainActor.run {
                    self.showError(error.localizedDescription)
                }
            }
        }
    }

    private func stopCapture() {
        guard isCapturing else { return }
        Task {
            await audioManager?.stopCapture()
            await MainActor.run {
                self.audioManager = nil
                self.isCapturing = false
                self.updateMenuBar(state: .ready)
                self.windowTracker?.stopTracking()
                self.overlayWindow?.hide()
            }
        }
    }

    // MARK: - Translation Pipeline

    private func handleAudioChunk(_ pcmData: Data) {
        translationClient?.translate(pcmData: pcmData) { [weak self] result in
            switch result {
            case .success(let response):
                guard !response.skipped,
                      !response.translation.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                DispatchQueue.main.async {
                    self?.overlayWindow?.showSubtitle(
                        response.translation,
                        mirroringWindowFrame: self?.windowTracker?.currentFrame
                    )
                }
            case .failure(let error):
                print("[AppDelegate] Translation error: \(error)")
            }
        }
    }

    // MARK: - Overlay Window

    private func setupOverlayWindow() {
        overlayWindow = SubtitleOverlayWindow()
        windowTracker = MirroringWindowTracker()
    }

    // MARK: - Error Display

    private func showError(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Hermes"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.runModal()
    }
}
```

---

## After writing both files

### Build the project

```bash
cd Hermes
xcodebuild -project Hermes.xcodeproj \
           -scheme Hermes \
           -configuration Debug \
           build \
           2>&1 | tail -30
```

If there are compile errors, fix them. Common issues:
- Missing `await` / `async` on ScreenCaptureKit calls
- `MirroringWindowTracker` not found → it's defined in SubtitleOverlayWindow.swift, verify that file
- `CaptureError` not found → it's defined in AudioCaptureManager.swift, verify that file

### Integration smoke test

1. Start the Python service manually: `cd translator_service && python main.py`
2. Wait 15s for models to load: `curl http://localhost:5005/health`
3. Open iPhone Mirroring on macOS
4. Run the built app
5. Click "Start Translating" in the menu bar
6. Play a video in any non-English language via iPhone Mirroring
7. Confirm subtitles appear within ~1 second

### Fix known edge cases

After the basic build passes, check and fix:

1. **Screen recording permission**: If the app crashes on `SCShareableContent.current`, add
   a pre-check with `SCShareableContent.info` and show a guidance dialog pointing to
   System Settings → Privacy → Screen Recording.

2. **Python path for packaged app**: The serviceDir path calculation assumes a specific
   directory structure. If the path is wrong, log the attempted path and adjust.

3. **Menu bar icon during loading**: Change the icon tint (template vs colored) to give
   visual feedback while the AI models are warming up.

Signal completion: "INTEGRATION COMPLETE — build succeeds and smoke test passes"
