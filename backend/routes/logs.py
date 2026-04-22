from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import os
import config

router = APIRouter()

async def log_generator():
    log_file = config.LOG_PATH
    
    # Wait until the log file is created
    while not os.path.exists(log_file):
        yield "data: Waiting for logs to appear...\n\n"
        await asyncio.sleep(2)
        
    with open(log_file, "r") as f:
        # Seek to the end of the file to only stream new logs
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            
            # Yield new lines as SSE
            yield f"data: {line.strip()}\n\n"

@router.get("/logs/stream")
async def stream_logs():
    """Stream live inference logs."""
    return StreamingResponse(log_generator(), media_type="text/event-stream")
