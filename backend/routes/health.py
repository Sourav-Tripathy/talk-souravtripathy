from fastapi import APIRouter
import psutil
import torch
from services.client_manager import IS_CPU

router = APIRouter()


@router.get("/health")
async def health():
    """Heartbeat endpoint — polled by the frontend every 2 minutes.

    Reports GPU stats when running on CUDA, or RAM/CPU stats when running in
    CPU-only mode (no CUDA device detected).
    """
    # ── CPU / RAM (always measured) ───────────────────────────────────────────
    cpu_percent = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    ram_used_gb  = round(vm.used  / 1e9, 2)
    ram_total_gb = round(vm.total / 1e9, 2)

    # ── RSS of this process ───────────────────────────────────────────────────
    proc_rss_mb = 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    proc_rss_mb = round(float(line.split()[1]) / 1024.0, 2)
                    break
    except Exception:
        pass

    # ── GPU stats (GPU mode only) ─────────────────────────────────────────────
    gpu_used_mb  = 0.0
    gpu_total_mb = 0.0
    if not IS_CPU and torch.cuda.is_available():
        gpu_used_mb  = round(torch.cuda.memory_allocated() / 1e6, 2)
        gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1e6, 2)

    return {
        "status":              "alive",
        "device":              "cpu" if IS_CPU else "gpu",
        # GPU fields (0 when in CPU mode)
        "gpu_memory_used_mb":  gpu_used_mb,
        "gpu_memory_total_mb": gpu_total_mb,
        # System stats (always present)
        "cpu_percent":         cpu_percent,
        "ram_used_gb":         ram_used_gb,
        "ram_total_gb":        ram_total_gb,
        # Process RSS (useful for CPU mode to see model footprint)
        "process_rss_mb":      proc_rss_mb,
    }
