import Foundation

/// Lightweight HTTP client for checking the Python translation service health.
/// The actual /translate calls are handled internally by AudioCaptureManager.
class TranslationClient {

    private let baseURL = URL(string: "http://127.0.0.1:5005")!
    private var pollTimer: Timer?

    // MARK: - Health Check

    /// Single health check. Returns true if the service responds 200 with {"status": "ready"}.
    func checkHealth() async -> Bool {
        let url = baseURL.appendingPathComponent("health")
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return false
            }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
               json["status"] == "ready" {
                return true
            }
            return false
        } catch {
            return false
        }
    }

    /// Poll /health at the given interval until the service is ready, then call onReady on the main queue.
    func pollUntilReady(interval: TimeInterval = 2.0, onReady: @escaping () -> Void) {
        // Perform the first check immediately
        Task {
            if await checkHealth() {
                await MainActor.run { onReady() }
                return
            }
            await MainActor.run { [weak self] in
                self?.startPolling(interval: interval, onReady: onReady)
            }
        }
    }

    /// Stop any active polling timer. Must be called on the main thread.
    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Private

    private func startPolling(interval: TimeInterval, onReady: @escaping () -> Void) {
        pollTimer?.invalidate()
        // Timer fires on the main run loop. We do the async health check inside,
        // and stop the timer via self.pollTimer reference (not the closure parameter)
        // to avoid capturing the non-Sendable Timer in a Task closure.
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            Task {
                let ready = await self.checkHealth()
                if ready {
                    await MainActor.run { [weak self] in
                        self?.stopPolling()
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
