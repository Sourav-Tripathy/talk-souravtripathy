from fastapi import FastAPI
from middleware.cors import add_cors
from routes.chat import router as chat_router
from routes.health import router as health_router
from routes.logs import router as logs_router
from services.vllm_client import init_engine
from services.mongo_service import init_mongo
import uvicorn
import config

app = FastAPI(
    title="talk.souravtripathy.com backend",
    description=(
        "FastAPI wrapper around a vLLM-served 500M parameter language model "
        "running on a home old GPU(NVDIA-GEFORCE-GTX-1650-4GB)"
    ),
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    init_mongo()
    init_engine()

add_cors(app)

app.include_router(chat_router)
app.include_router(health_router)
app.include_router(logs_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=False,       # reload must be False — vLLM engine for some reason crashing on reload
        log_level="info",
    )


# TODO: Think of a way in which model load to VRAM is done when servere starts but infreence engine comes from frontend configurable something like that..That means implementation of other inference methods like llama.cpp or direct transformer or Tensor RT or ONNX runtime or SGLang.