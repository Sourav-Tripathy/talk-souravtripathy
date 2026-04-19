from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import config

# Motor client is a module-level singleton (thread-safe, async-safe).
_client: AsyncIOMotorClient | None = None


def _get_collection():
    global _client
    if _client is None:
        if not config.MONGODB_URI:
            raise RuntimeError("MONGODB_URI is not set in the environment.")
        _client = AsyncIOMotorClient(config.MONGODB_URI)
    return _client[config.DB_NAME][config.COLLECTION_NAME]


async def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Persist a single conversation turn to MongoDB Atlas."""
    collection = _get_collection()
    await collection.insert_one(
        {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc),
            "user": user_msg,
            "assistant": assistant_msg,
        }
    )
