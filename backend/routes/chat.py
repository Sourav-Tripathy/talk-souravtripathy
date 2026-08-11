from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.client_manager import generate_stream, IS_CPU
from services.logger import log_inference
import json
import asyncio
import time

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str


@router.post("/chat")
async def chat(req: ChatRequest):
    """Main inference endpoint — streams the response text using SSE."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def stream_generator():
        full_text = ""
        metrics = None
        start_time = time.perf_counter()

        system_message = "You are a helpful, respectful, and honest AI assistant. Always answer as helpfully as possible, while being concise."
        formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n"
        
        async for chunk in generate_stream(formatted_prompt):
            if chunk["type"] == "token":
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "metrics":
                full_text = chunk["full_text"]
                metrics = chunk["content"]
                metrics["elapsed_time"] = round(time.perf_counter() - start_time, 2)
                yield f"data: {json.dumps(chunk)}\n\n"

        log_inference(req.message, full_text, metrics)

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
