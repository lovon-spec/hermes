#!/usr/bin/env python3
"""
Georgian STT worker -- persistent subprocess using NeMo's cache-aware streaming.

Uses the encoder's native cache state to maintain context across audio chunks,
giving dramatically better transcription quality compared to independent
per-chunk batch transcription.

Protocol (unchanged):
  stdin:  4-byte big-endian length prefix, then that many PCM bytes (16kHz mono 16-bit LE)
  stdout: JSON line per result: {"text": "...", "language": "ka"}
  stderr: all logging / NeMo output
  First stdout line: {"status": "ready"}

Special: length == 0 resets streaming state (cache + accumulated predictions).
"""

import json
import os
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Streaming state -- kept across chunks, reset on empty-length message
# ---------------------------------------------------------------------------
_cache_last_channel = None
_cache_last_time = None
_cache_last_channel_len = None
_prev_pred_out = None
_prev_full_text = ""


def _reset_state(model, device):
    """Reset all streaming state to initial values."""
    global _cache_last_channel, _cache_last_time, _cache_last_channel_len
    global _prev_pred_out, _prev_full_text

    _cache_last_channel, _cache_last_time, _cache_last_channel_len = (
        model.encoder.get_initial_cache_state(
            batch_size=1, dtype=torch.float32, device=device
        )
    )
    _prev_pred_out = None
    _prev_full_text = ""


def _extract_text(transcriptions):
    """Extract text string from conformer_stream_step transcription output.

    In CTC mode the output is a list of Hypothesis objects (one per batch
    item) each having a .text attribute.  We handle multiple possible
    return shapes defensively.
    """
    if not transcriptions:
        return ""
    item = transcriptions[0]
    if hasattr(item, "text"):
        return str(item.text).strip()
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (list, tuple)) and len(item) > 0:
        inner = item[0]
        return str(inner.text if hasattr(inner, "text") else inner).strip()
    return str(item).strip()


def main():
    global _cache_last_channel, _cache_last_time, _cache_last_channel_len
    global _prev_pred_out, _prev_full_text
    global torch  # make torch available to _reset_state

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # ---- Redirect all logging to stderr so stdout stays JSON-only ----------
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    for name in [
        "pytorch_lightning", "nemo_logger", "nemo", "torch", "matplotlib",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import torch as _torch
    torch = _torch  # module-level reference for _reset_state
    torch.set_num_threads(os.cpu_count() or 8)

    # Suppress NeMo's print() calls during import and model load
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    sys.stderr.write("Georgian worker: loading model...\n")
    sys.stderr.flush()

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name="nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()

    # Switch to CTC decoder with greedy_batch (faster than default greedy)
    decoding_cfg = OmegaConf.create({"strategy": "greedy_batch"})
    model.change_decoding_strategy(decoder_type="ctc", decoding_cfg=decoding_cfg)

    # ---- Initialise streaming state ----------------------------------------
    _reset_state(model, device)

    # ---- Warmup: run one chunk of silence through the streaming path -------
    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    silence_tensor = torch.from_numpy(silence).unsqueeze(0).to(device)
    silence_length = torch.tensor([len(silence)], dtype=torch.long, device=device)
    with torch.no_grad():
        features, feat_length = model.preprocessor(
            input_signal=silence_tensor, length=silence_length
        )
        model.conformer_stream_step(
            processed_signal=features,
            processed_signal_length=feat_length,
            cache_last_channel=_cache_last_channel,
            cache_last_time=_cache_last_time,
            cache_last_channel_len=_cache_last_channel_len,
            keep_all_outputs=True,
            previous_pred_out=None,
            return_transcription=True,
        )
    # Reset after warmup so real audio starts fresh
    _reset_state(model, device)

    sys.stdout = real_stdout
    sys.stderr.write(f"Georgian worker: ready on {device}\n")
    sys.stderr.flush()

    # ---- Signal ready to parent process ------------------------------------
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    # ---- Main loop: read length-prefixed PCM, stream-transcribe, emit JSON -
    while True:
        try:
            header = sys.stdin.buffer.read(4)
            if len(header) < 4:
                break  # parent closed pipe

            length = struct.unpack(">I", header)[0]

            # Length == 0 is the reset signal
            if length == 0:
                _reset_state(model, device)
                sys.stdout.write(json.dumps({"text": "", "language": "ka"}) + "\n")
                sys.stdout.flush()
                continue

            pcm_bytes = sys.stdin.buffer.read(length)
            if len(pcm_bytes) < length:
                break  # truncated read -- parent gone

            # Convert PCM int16 LE -> float32 normalised to [-1, 1]
            int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio = int16.astype(np.float32) / 32768.0

            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
            audio_length = torch.tensor(
                [len(audio)], dtype=torch.long, device=device
            )

            t0 = time.perf_counter()

            # Redirect stdout during inference (NeMo may print warnings)
            sys.stdout = sys.stderr
            with torch.no_grad():
                # Step 1: compute mel spectrogram features
                features, feat_length = model.preprocessor(
                    input_signal=audio_tensor, length=audio_length
                )

                # Step 2: cache-aware streaming encoder + CTC decode
                result = model.conformer_stream_step(
                    processed_signal=features,
                    processed_signal_length=feat_length,
                    cache_last_channel=_cache_last_channel,
                    cache_last_time=_cache_last_time,
                    cache_last_channel_len=_cache_last_channel_len,
                    keep_all_outputs=True,
                    previous_pred_out=_prev_pred_out,
                    return_transcription=True,
                )

                (
                    greedy_preds,
                    transcriptions,
                    _cache_last_channel,
                    _cache_last_time,
                    _cache_last_channel_len,
                    _best_hyp,
                ) = result

                # Update accumulated prediction tokens for next chunk
                _prev_pred_out = greedy_preds

            sys.stdout = real_stdout
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Extract the full accumulated transcription
            full_text = _extract_text(transcriptions)

            # Diff against previous full text to find newly added content.
            # The CTC decoder re-decodes the full accumulated prediction
            # sequence each time, so full_text grows monotonically.
            if full_text.startswith(_prev_full_text):
                new_text = full_text[len(_prev_full_text):].strip()
            else:
                # If the full text diverged (CTC re-decoding can revise
                # earlier tokens), emit the full new text this chunk.
                new_text = full_text.strip()

            _prev_full_text = full_text

            sys.stderr.write(
                f"Georgian worker: chunk {len(pcm_bytes)}B -> "
                f"'{new_text}' ({elapsed_ms:.0f}ms)\n"
            )
            sys.stderr.flush()

            sys.stdout.write(
                json.dumps({"text": new_text, "language": "ka"}) + "\n"
            )
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write(f"Georgian worker error: {e}\n")
            sys.stderr.flush()
            # Recover stdout in case it was redirected
            sys.stdout = real_stdout
            sys.stdout.write(json.dumps({"text": "", "language": "ka"}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
