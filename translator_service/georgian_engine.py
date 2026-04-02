"""
Georgian STT engine using NVIDIA NeMo FastConformer.

Runs NeMo in a persistent subprocess (separate venv) to avoid
dependency conflicts with transformers/huggingface_hub. The worker
loads the model once and processes chunks via stdin/stdout.

Model: nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc
  - 115M parameters, ~460MB, 7.44% WER on Georgian
"""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import sys
import threading

logger = logging.getLogger("hermes.georgian")

_VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv-nemo")
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "georgian_worker.py")

_lock = threading.Lock()
_process = None
_inference_lock = threading.Lock()


def _get_python():
    venv_python = os.path.join(_VENV_DIR, "bin", "python3")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable


def _ensure_worker():
    """Start the persistent worker subprocess if not running."""
    global _process
    if _process is not None and _process.poll() is None:
        return True  # already running

    python = _get_python()
    if not os.path.isfile(python):
        logger.warning("NeMo venv not found at %s", _VENV_DIR)
        return False

    logger.info("Starting Georgian worker subprocess: %s", python)
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    _process = subprocess.Popen(
        [python, _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Log worker stderr in background
    import threading
    def _drain_stderr():
        for line in _process.stderr:
            logger.info("[Georgian worker] %s", line.decode().rstrip())
    threading.Thread(target=_drain_stderr, daemon=True).start()

    # Wait for "ready" signal — skip any non-JSON lines NeMo might print to stdout
    try:
        import time
        deadline = time.time() + 120  # 2 minute timeout for model download + load
        while time.time() < deadline:
            line = _process.stdout.readline().decode().strip()
            if not line:
                if _process.poll() is not None:
                    logger.error("Georgian worker exited during startup")
                    return False
                continue
            try:
                status = json.loads(line)
                if status.get("status") == "ready":
                    logger.info("Georgian worker ready (pid %d)", _process.pid)
                    return True
            except json.JSONDecodeError:
                logger.info("[Georgian worker stdout] %s", line)
                continue
    except Exception as e:
        logger.error("Georgian worker failed to start: %s", e)

    return False


def transcribe(pcm_bytes: bytes) -> dict:
    """Transcribe raw PCM audio to Georgian text via persistent subprocess."""
    if not pcm_bytes or len(pcm_bytes) < 1600:
        return {"text": "", "language": "ka"}

    global _process
    with _inference_lock:
        with _lock:
            if not _ensure_worker():
                return {"text": "", "language": "ka"}

        try:
            # Send length-prefixed PCM
            header = struct.pack(">I", len(pcm_bytes))
            _process.stdin.write(header)
            _process.stdin.write(pcm_bytes)
            _process.stdin.flush()

            # Read JSON line response
            line = _process.stdout.readline().decode().strip()
            if not line:
                logger.error("Georgian worker returned empty response")
                return {"text": "", "language": "ka"}

            return json.loads(line)
        except (BrokenPipeError, OSError) as e:
            logger.error("Georgian worker pipe error: %s", e)
            with _lock:
                _process = None
            return {"text": "", "language": "ka"}


def warmup() -> None:
    """Start the worker and load the model."""
    with _lock:
        if _ensure_worker():
            logger.info("Georgian engine warmed up")
        else:
            logger.warning("Georgian engine failed to start — Georgian STT disabled")


def shutdown():
    """Stop the worker subprocess."""
    global _process
    if _process is not None:
        _process.stdin.close()
        _process.terminate()
        _process = None
