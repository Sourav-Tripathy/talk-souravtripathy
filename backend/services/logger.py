import logging
import subprocess
import psutil
import torch
from datetime import datetime, timezone
import config
import os

# Ensure the logs directory exists before the handler opens the file.
os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_PATH,
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("inference")


def _get_gpu_util() -> int:
    """Query nvidia-smi for current GPU utilisation percentage. Returns -1 on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def log_inference(prompt: str, response: str, metrics: dict) -> None:
    """
    Write a structured log entry after every inference call.

    The latency_ms and vram_used_mb fields here are the raw material for
    the Phase 1→2 comparison once custom kernels are added.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_preview": prompt[:80],
        "response_preview": response[:80],
        "latency_ms": metrics.get("latency_ms"),
        "vram_used_mb": metrics.get("vram_used_mb"),
        "vram_delta_mb": metrics.get("vram_delta_mb"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "gpu_util_percent": _get_gpu_util(),
    }
    logger.info(str(entry))
