from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import config

# Motor client is a module-level singleton (thread-safe, async-safe).
_client: AsyncIOMotorClient | None = None

def init_mongo():
    global _client
    if not config.MONGODB_URI:
        print("Warning: MONGODB_URI is not set in the environment. Chat history won't be saved.")
        return
    _client = AsyncIOMotorClient(config.MONGODB_URI)

def _get_collection():
    if _client is None:
        raise RuntimeError("MongoDB client is not initialized.")
    return _client[config.DB_NAME][config.COLLECTION_NAME]


async def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Persist a single conversation turn to MongoDB Atlas."""
    if _client is None:
        return
    try:
        collection = _get_collection()
        await collection.insert_one(
            {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc),
                "user": user_msg,
                "assistant": assistant_msg,
            }
        )
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
