import Foundation

/// Lightweight HTTP client for checking the Python translation service health.
class TranslationClient {

    private let baseURL = URL(string: "http://127.0.0.1:5005")!
    private var pollTimer: Timer?
    private let logFile = "/tmp/hermes-debug.log"

    private func log(_ message: String) {
        let line = "\(Date()): [TranslationClient] \(message)\n"
        if let data = line.data(using: .utf8),
           let handle = FileHandle(forWritingAtPath: logFile) {
            handle.seekToEndOfFile()
            handle.write(data)
            handle.closeFile()
        }
    }

    func checkHealth() async -> Bool {
        let url = baseURL.appendingPathComponent("health")
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse else {
                log("health: no HTTP response")
                return false
            }
            log("health: HTTP \(httpResponse.statusCode)")
            guard httpResponse.statusCode == 200 else { return false }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
               json["status"] == "ready" {
                log("health: READY")
                return true
            }
            return false
        } catch {
            log("health: error \(error.localizedDescription)")
            return false
        }
    }

    func pollUntilReady(interval: TimeInterval = 2.0, onReady: @escaping () -> Void) {
        log("pollUntilReady started")
        startPolling(interval: interval, onReady: onReady)
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func startPolling(interval: TimeInterval, onReady: @escaping () -> Void) {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.log("poll tick")
            Task {
                let ready = await self.checkHealth()
                if ready {
                    await MainActor.run {
                        self.log("calling onReady callback")
                        self.stopPolling()
                        onReady()
                    }
                }
            }
        }
    }

    deinit {
        pollTimer?.invalidate()
        pollTimer = nil
    }
}
