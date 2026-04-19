"""
CORS middleware configuration.

During development, add localhost origins to config.ALLOWED_ORIGINS.
In production only talk.souravtripathy.com is allowed.
"""
from fastapi.middleware.cors import CORSMiddleware
from config import ALLOWED_ORIGINS


def add_cors(app) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
