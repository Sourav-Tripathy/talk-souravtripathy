from fastapi import APIRouter
import psutil
import torch

router = APIRouter()

@router.get("/health")
async def health():
    """Heartbeat endpoint — polled by the frontend every 2 minutes."""
    gpu_used_mb = 0.0
    gpu_total_mb = 0.0
    if torch.cuda.is_available():
        gpu_used_mb = round(torch.cuda.memory_allocated() / 1e6, 2)
        gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1e6, 2)

    return {
        "status": "alive",
        "gpu_memory_used_mb": gpu_used_mb,
        "gpu_memory_total_mb": gpu_total_mb,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
    }
