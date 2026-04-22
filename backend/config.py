import os
from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", "./model")
MAX_MODEL_LEN: int = 2048
GPU_MEMORY_UTILIZATION: float = 0.80
DTYPE: str = "float16"

# ── Inference defaults ────────────────────────────────────────────────────────
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 512
TOP_P: float = 0.9

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
DB_NAME: str = "talk_sourav"
COLLECTION_NAME: str = "conversations"

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
