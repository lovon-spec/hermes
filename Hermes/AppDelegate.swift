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
