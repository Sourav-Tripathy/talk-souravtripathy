from fastapi import FastAPI
from middleware.cors import add_cors
from routes.chat import router as chat_router
from routes.health import router as health_router
from routes.logs import router as logs_router
from services.client_manager import init_engine, IS_CPU
import uvicorn
import config

app = FastAPI(
    title="talk.souravtripathy.com backend",
    description=(
        "FastAPI wrapper around various inference engines running on a local device"
    ),
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    init_engine()

add_cors(app)

app.include_router(chat_router)
app.include_router(health_router)
app.include_router(logs_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,       # reload must be False — vLLM engine for some reason crashing on reload
        log_level="info",
    )