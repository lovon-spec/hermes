---
name: swift-audio-capture
description: Implements AudioCaptureManager.swift using ScreenCaptureKit to capture audio from the iPhone Mirroring window. Converts CMSampleBuffers to 16kHz mono PCM chunks and sends them to the Python microservice. Run after swift-scaffold completes. Independent of swift-ui-overlay.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
effort: max
background: true
---

You are implementing `Hermes/AudioCaptureManager.swift`.

Read the existing stub first, then replace it with the full implementation below.

## Full implementation

```swift
import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

/// Callback type: receives raw 16kHz mono PCM bytes ready to send to the translation service
typealias AudioChunkCallback = (Data) -> Void

class AudioCaptureManager: NSObject {

    // MARK: - Configuration
    private let targetSampleRate: Double = 16000
    private let chunkDuration: Double = 2.0       // seconds per chunk sent to Whisper
    private let overlapDuration: Double = 0.5     // overlap between chunks to avoid word cuts

    // MARK: - State
    private var stream: SCStream?
    private var audioConverter: AVAudioConverter?
    private var pcmBuffer = Data()
    private var lastSentCutoff = 0               // byte offset of last sent chunk end
    private let bufferQueue = DispatchQueue(label: "com.hermes.translator.audio")
    var onChunkReady: AudioChunkCallback?

    // MARK: - Start / Stop

    func startCapture() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        // Find iPhone Mirroring window dynamically — its display name contains "iPhone" or the device name
        guard let iphoneApp = content.applications.first(where: { app in
            app.bundleIdentifier == "com.apple.ScreenContinuity" ||
            app.applicationName.localizedCaseInsensitiveContains("iPhone Mirroring") ||
            app.applicationName.localizedCaseInsensitiveContains("iPhone")
        }) else {
            throw CaptureError.iphoneMirroringNotFound
        }

        let filter = SCContentFilter(desktopIndependentWindow: content.windows.first(where: {
            $0.owningApplication?.bundleIdentifier == iphoneApp.bundleIdentifier
        }) ?? content.windows[0])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000        // capture at native rate, we downsample
        config.channelCount = 1

        stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream?.addStreamOutput(self, type: .audio, sampleHandlerQueue: bufferQueue)
        try await stream?.startCapture()
    }

    func stopCapture() async {
        try? await stream?.stopCapture()
        stream = nil
        bufferQueue.sync { self.pcmBuffer = Data() }
    }

    // MARK: - Audio Processing

    /// Convert an SCStream audio buffer to 16kHz mono PCM Data
    private func processAudioBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)!.pointee

        // Set up converter once (48kHz stereo → 16kHz mono)
        if audioConverter == nil {
            let inputFormat = AVAudioFormat(streamDescription: &CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)!.pointee)!
            let outputFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                             sampleRate: targetSampleRate,
                                             channels: 1,
                                             interleaved: true)!
            audioConverter = AVAudioConverter(from: inputFormat, to: outputFormat)
        }
        guard let converter = audioConverter else { return }

        // Get raw bytes from block buffer
        var dataPointer: UnsafeMutablePointer<Int8>?
        var dataLength = 0
        CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
                                    totalLengthOut: &dataLength, dataPointerOut: &dataPointer)
        guard let ptr = dataPointer else { return }

        let frameCount = CMSampleBufferGetNumSamples(sampleBuffer)
        let inputFormat = converter.inputFormat
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * targetSampleRate / asbd.mSampleRate)

        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(frameCount)),
              let outputBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: outputFrameCount) else { return }

        inputBuffer.frameLength = AVAudioFrameCount(frameCount)
        memcpy(inputBuffer.audioBufferList.pointee.mBuffers.mData, ptr, dataLength)

        var convError: NSError?
        converter.convert(to: outputBuffer, error: &convError) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error = convError { print("[AudioCapture] Conversion error: \(error)"); return }

        outputBuffer.frameLength = outputFrameCount
        let int16Ptr = outputBuffer.int16ChannelData![0]
        let byteCount = Int(outputBuffer.frameLength) * 2  // 2 bytes per Int16 sample
        pcmBuffer.append(Data(bytes: int16Ptr, count: byteCount))

        flushChunkIfReady()
    }

    private func flushChunkIfReady() {
        let chunkBytes = Int(targetSampleRate * chunkDuration) * 2   // 2 bytes/sample
        let overlapBytes = Int(targetSampleRate * overlapDuration) * 2

        while pcmBuffer.count - lastSentCutoff >= chunkBytes {
            let start = max(0, lastSentCutoff - overlapBytes)
            let end = lastSentCutoff + chunkBytes
            let chunk = pcmBuffer.subdata(in: start..<min(end, pcmBuffer.count))
            onChunkReady?(chunk)
            lastSentCutoff += chunkBytes

            // Trim buffer to avoid unbounded growth — keep last 10 seconds
            let maxBytes = Int(targetSampleRate * 10) * 2
            if pcmBuffer.count > maxBytes {
                let trim = pcmBuffer.count - maxBytes
                pcmBuffer = pcmBuffer.dropFirst(trim)
                lastSentCutoff = max(0, lastSentCutoff - trim)
            }
        }
    }
}

// MARK: - SCStreamDelegate

extension AudioCaptureManager: SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        print("[AudioCapture] Stream stopped: \(error.localizedDescription)")
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
```

## After writing the file

Verify it compiles by running from the project root:
```bash
swiftc -typecheck Hermes/AudioCaptureManager.swift \
  -sdk $(xcrun --show-sdk-path) \
  -target arm64-apple-macos15.0 \
  -framework ScreenCaptureKit \
  -framework AVFoundation \
  2>&1
```

Fix any type errors before completing. The file must compile cleanly.

Signal completion: "AUDIO CAPTURE COMPLETE"
