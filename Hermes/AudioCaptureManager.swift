import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

/// Callback type: receives raw 16kHz mono PCM bytes ready to send to the translation service
typealias AudioChunkCallback = (Data) -> Void

class AudioCaptureManager: NSObject {

    // MARK: - Debug logging
    private let logFile = "/tmp/hermes-debug.log"
    private func log(_ message: String) {
        let line = "\(Date()): [Audio] \(message)\n"
        if let data = line.data(using: .utf8),
           let handle = FileHandle(forWritingAtPath: logFile) {
            handle.seekToEndOfFile()
            handle.write(data)
            handle.closeFile()
        }
    }

    // MARK: - Configuration
    private let targetSampleRate: Double = 16000
    private let chunkDuration: Double = 3.0       // seconds per chunk sent to Whisper
    private let overlapDuration: Double = 0.3     // small overlap to catch boundary words
    private let translationURL = URL(string: "http://localhost:5005/translate")!

    // MARK: - State
    private var stream: SCStream?
    private var pcmBuffer = Data()
    private let bufferQueue = DispatchQueue(label: "com.hermes.translator.audio")
    private(set) var isCapturing = false

    /// Called on the main queue with (translation, originalText?)
    var onTranslation: ((String, String?) -> Void)?

    /// Called with raw PCM chunk data (for external consumers, if needed)
    var onChunkReady: AudioChunkCallback?

    // MARK: - Start / Stop

    func startCapture() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        // Find iPhone Mirroring window dynamically -- its display name contains "iPhone"
        // or the app uses the known ScreenContinuity bundle identifier
        // Log all available apps for debugging
        log("Available apps: \(content.applications.map { "\($0.applicationName) (\($0.bundleIdentifier))" })")

        guard let iphoneApp = content.applications.first(where: { app in
            app.bundleIdentifier == "com.apple.ScreenContinuity" ||
            app.applicationName.localizedCaseInsensitiveContains("iPhone Mirroring") ||
            app.applicationName.localizedCaseInsensitiveContains("iPhone")
        }) else {
            log("ERROR: iPhone Mirroring app not found in \(content.applications.count) apps")
            throw CaptureError.iphoneMirroringNotFound
        }
        log("Found iPhone Mirroring app: \(iphoneApp.applicationName) (\(iphoneApp.bundleIdentifier))")

        let targetWindow = content.windows.first(where: {
            $0.owningApplication?.bundleIdentifier == iphoneApp.bundleIdentifier
        })
        guard let window = targetWindow else {
            log("ERROR: No window found for \(iphoneApp.bundleIdentifier)")
            throw CaptureError.iphoneMirroringNotFound
        }
        log("Target window: \(window.title ?? "untitled") frame=\(window.frame)")

        let filter = SCContentFilter(desktopIndependentWindow: window)

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000        // capture at native rate, we downsample in processAudioBuffer
        config.channelCount = 1
        // Minimize video overhead -- we only need audio
        config.width = 2
        config.height = 2

        let newStream = SCStream(filter: filter, configuration: config, delegate: self)
        try newStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: bufferQueue)
        try await newStream.startCapture()

        stream = newStream
        isCapturing = true
        log("Capture started successfully")
    }

    func stopCapture() async {
        try? await stream?.stopCapture()
        stream = nil
        isCapturing = false
        bufferQueue.sync {
            self.pcmBuffer = Data()
        }
    }

    // MARK: - Audio Processing

    private var audioCallbackCount = 0
    /// Convert an SCStream audio buffer to 16kHz mono PCM Data and accumulate into the buffer
    private func processAudioBuffer(_ sampleBuffer: CMSampleBuffer) {
        audioCallbackCount += 1
        if audioCallbackCount <= 3 || audioCallbackCount % 50 == 0 {
            log("processAudioBuffer #\(audioCallbackCount), buffer=\(pcmBuffer.count) bytes")
        }
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }

        let sourceSampleRate = asbd.pointee.mSampleRate
        let frameCount = CMSampleBufferGetNumSamples(sampleBuffer)
        guard frameCount > 0 else { return }

        // Extract raw bytes from the sample buffer
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var dataLength = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0,
                                                  lengthAtOffsetOut: nil,
                                                  totalLengthOut: &dataLength,
                                                  dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer, dataLength > 0 else { return }

        // Source format from ScreenCaptureKit (typically 48kHz Float32 mono)
        var sourceASBD = asbd.pointee
        let sourceFormat = AVAudioFormat(streamDescription: &sourceASBD)!

        // Target: 16kHz, 16-bit signed integer, mono, interleaved
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                         sampleRate: targetSampleRate,
                                         channels: 1,
                                         interleaved: true)!

        // Create input buffer from raw bytes
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat,
                                                  frameCapacity: AVAudioFrameCount(frameCount)) else { return }
        inputBuffer.frameLength = AVAudioFrameCount(frameCount)
        memcpy(inputBuffer.audioBufferList.pointee.mBuffers.mData, ptr, dataLength)

        // If source is already 16kHz Int16, just append directly
        if sourceSampleRate == targetSampleRate && sourceFormat.commonFormat == AVAudioCommonFormat.pcmFormatInt16 {
            pcmBuffer.append(Data(bytes: ptr, count: dataLength))
            flushChunkIfReady()
            return
        }

        // Set up converter: source format -> 16kHz Int16 mono
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else { return }

        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * targetSampleRate / sourceSampleRate)
        guard outputFrameCount > 0 else { return }
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                   frameCapacity: outputFrameCount) else { return }

        var conversionError: NSError?
        var inputConsumed = false
        converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if inputConsumed {
                outStatus.pointee = AVAudioConverterInputStatus.noDataNow
                return nil
            }
            inputConsumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error = conversionError {
            print("[AudioCapture] Conversion error: \(error)")
            return
        }

        guard outputBuffer.frameLength > 0 else { return }

        // Append 16-bit PCM samples to our rolling buffer
        let int16Ptr = outputBuffer.int16ChannelData![0]
        let byteCount = Int(outputBuffer.frameLength) * 2  // 2 bytes per Int16 sample
        pcmBuffer.append(Data(bytes: int16Ptr, count: byteCount))

        flushChunkIfReady()
    }

    /// When enough audio has accumulated, flush a chunk (with overlap) and send to translation service
    private func flushChunkIfReady() {
        let chunkBytes = Int(targetSampleRate * chunkDuration) * 2   // 2 bytes per sample
        let overlapBytes = Int(targetSampleRate * overlapDuration) * 2

        while pcmBuffer.count >= chunkBytes {
            let chunk = pcmBuffer.prefix(chunkBytes)

            // Compute RMS to skip silence
            let rms = chunk.withUnsafeBytes { buf -> Double in
                let int16s = buf.bindMemory(to: Int16.self)
                var sum: Double = 0
                for s in int16s { sum += Double(s) * Double(s) }
                return sqrt(sum / Double(int16s.count))
            }

            // Consume chunk but keep last overlapBytes for next chunk's start
            let advance = chunkBytes - overlapBytes
            pcmBuffer = pcmBuffer.subdata(in: advance..<pcmBuffer.count)

            if rms < 50 {
                log("Skipping silent chunk (RMS=\(String(format: "%.1f", rms)))")
                continue
            }

            log("Sending chunk: \(chunk.count) bytes (\(Double(chunk.count) / 32000.0)s audio) RMS=\(String(format: "%.1f", rms))")
            onChunkReady?(Data(chunk))
            sendChunkToTranslationService(Data(chunk))
        }
    }

    // MARK: - Translation Service Communication

    /// POST raw PCM bytes to the local Python translation service
    private func sendChunkToTranslationService(_ audioData: Data) {
        var request = URLRequest(url: translationURL)
        request.httpMethod = "POST"
        request.setValue("audio/raw", forHTTPHeaderField: "Content-Type")
        request.httpBody = audioData
        request.timeoutInterval = 30

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                self?.log("Translation request FAILED: \(error.localizedDescription)")
                return
            }
            guard let data = data else {
                self?.log("Translation response: no data")
                return
            }

            if let raw = String(data: data, encoding: .utf8) {
                self?.log("Translation response: \(raw.prefix(200))")
            }

            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let translation = json["translation"] as? String,
                   !translation.isEmpty {
                    let original = json["original_text"] as? String
                    let skipped = json["skipped"] as? Bool ?? true
                    self?.log("Showing subtitle: \(translation)")
                    DispatchQueue.main.async {
                        // Pass original text only for actual translations (not skipped/passthrough)
                        self?.onTranslation?(translation, skipped ? nil : original)
                    }
                }
            } catch {
                self?.log("Failed to parse response: \(error)")
            }
        }.resume()
    }
}

// MARK: - SCStreamDelegate

extension AudioCaptureManager: SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("Stream stopped with error: \(error.localizedDescription)")
        DispatchQueue.main.async { [weak self] in
            self?.isCapturing = false
        }
    }
}

// MARK: - SCStreamOutput

extension AudioCaptureManager: SCStreamOutput {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of outputType: SCStreamOutputType) {
        guard outputType == .audio else { return }
        processAudioBuffer(sampleBuffer)
    }
}

// MARK: - Errors

enum CaptureError: LocalizedError {
    case iphoneMirroringNotFound

    var errorDescription: String? {
        switch self {
        case .iphoneMirroringNotFound:
            return "iPhone Mirroring is not running. Please open iPhone Mirroring and try again."
        }
    }
}
