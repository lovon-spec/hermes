#!/usr/bin/env python3
"""
Georgian STT worker — persistent subprocess running inside the NeMo venv.

Protocol: reads 4-byte big-endian length prefix, then that many PCM bytes.
Writes JSON line to stdout for each result. Loops forever.
"""

import json
import os
import struct
import sys
import tempfile
import wave

import numpy as np


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Redirect ALL logging to stderr so stdout stays clean for our JSON protocol
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    for name in ["pytorch_lightning", "nemo_logger", "nemo", "torch", "matplotlib"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import torch
    torch.set_num_threads(os.cpu_count() or 8)

    # Suppress NeMo's own print() calls by temporarily redirecting stdout during import
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    import nemo.collections.asr as nemo_asr
    sys.stdout = real_stdout

    # Load model once — redirect stdout to stderr during loading (NeMo prints to stdout)
    sys.stderr.write("Georgian worker: loading model...\n")
    sys.stderr.flush()
    sys.stdout = sys.stderr
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name="nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    model.change_decoding_strategy(decoder_type="ctc")

    # Warmup
    silence_path = _write_wav(np.zeros(8000, dtype=np.int16))
    with torch.no_grad():
        model.transcribe([silence_path])
    os.unlink(silence_path)
    sys.stdout = real_stdout

    sys.stderr.write(f"Georgian worker: ready on {device}\n")
    sys.stderr.flush()

    sys.stderr.write("Georgian worker: ready\n")
    sys.stderr.flush()

    # Signal ready
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    # Main loop: read length-prefixed PCM, transcribe, write JSON
    while True:
        try:
            header = sys.stdin.buffer.read(4)
            if len(header) < 4:
                break  # parent closed pipe

            length = struct.unpack(">I", header)[0]
            if length == 0:
                sys.stdout.write(json.dumps({"text": "", "language": "ka"}) + "\n")
                sys.stdout.flush()
                continue

            pcm_bytes = sys.stdin.buffer.read(length)
            if len(pcm_bytes) < length:
                break  # truncated read = parent gone

            int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            wav_path = _write_wav(int16)
            try:
                # Redirect stdout during transcribe (NeMo prints progress)
                sys.stdout = sys.stderr
                with torch.no_grad():
                    results = model.transcribe([wav_path])
                sys.stdout = real_stdout

                if isinstance(results, tuple):
                    results = results[0]

                text = ""
                if results and len(results) > 0:
                    text = results[0]
                    if hasattr(text, "text"):
                        text = text.text
                    text = str(text).strip()

                sys.stdout.write(json.dumps({"text": text, "language": "ka"}) + "\n")
                sys.stdout.flush()
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        except Exception as e:
            sys.stderr.write(f"Georgian worker error: {e}\n")
            sys.stderr.flush()
            sys.stdout.write(json.dumps({"text": "", "language": "ka"}) + "\n")
            sys.stdout.flush()


def _write_wav(int16_audio: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_audio.tobytes())
    os.close(fd)
    return path


if __name__ == "__main__":
    main()
