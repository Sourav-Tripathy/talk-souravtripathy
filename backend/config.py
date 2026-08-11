import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ── Device Detection ─────────────────────────────────────────────────────────
# True when no CUDA GPU is available — engine runs on CPU instead.
IS_CPU: bool = not torch.cuda.is_available()

# ── Engine Selection ─────────────────────────────────────────────────────────
ENGINE_TYPE: str = os.getenv("ENGINE_TYPE", "vllm").lower()
if ENGINE_TYPE not in ("vllm", "sglang", "tensorrt", "transformer"):
    ENGINE_TYPE = "vllm"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", "./model")
MAX_MODEL_LEN: int = 2048
GPU_MEMORY_UTILIZATION: float = 0.80
# On CPU, float16 is unstable in PyTorch; bfloat16 is the recommended dtype.
# On GPU, float16 is used for speed on older cards (e.g. GTX 1650).
DTYPE: str = "bfloat16" if IS_CPU else "float16"

# ── CPU-specific settings ─────────────────────────────────────────────────────
# KV-cache budget (GiB) reserved for CPU inference.
# Larger values allow more concurrent requests / longer contexts but consume RAM.
# Maps directly to vLLM's VLLM_CPU_KVCACHE_SPACE env variable.
# Raise to 8 or 16 if you have enough RAM (check with: free -h).
CPU_KVCACHE_SPACE: int = 4  # GiB

# Propagate the value into the environment right here so that any subsequent
# import of vllm or its C++ layer sees it before the engine process forks.
# An explicit env-var override still wins (set before launching the server).
if IS_CPU:
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", str(CPU_KVCACHE_SPACE))
    # Tells vLLM to use the CPU execution backend (not a kwarg on AsyncEngineArgs).
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    
    if ENGINE_TYPE == "sglang":
        # Enable SGLang CPU engine
        os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")


# ── Inference defaults ────────────────────────────────────────────────────────
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 1000
TOP_P: float = 0.9
# ── Server ────────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "5000"))
ALLOWED_ORIGINS: list[str] = [
    "https://talk.souravtripathy.com",
    # uncomment during local dev:
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH: str = "logs/inference.log"
