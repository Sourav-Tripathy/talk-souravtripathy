import config

IS_CPU: bool = config.IS_CPU

def init_engine():
    """Initialize the TensorRT-LLM engine (stub)."""
    print(f"[TENSORRT_LLM] Initializing engine (stub)...")

async def generate_stream(prompt: str):
    """Placeholder generation stream for TensorRT-LLM."""
    print(f"[TENSORRT_LLM] Generating stream (stub) for prompt: {prompt[:50]}...")
    yield {"type": "token", "content": f"TensorRT-LLM stub response for prompt: {prompt}"}
    yield {
        "type": "metrics",
        "content": {
            "device": "cpu" if IS_CPU else "gpu",
            "latency_ms": 10.0,
            "vram_delta_mb": 0.0,
            "vram_used_mb": 0.0,
            "ram_delta_mb": 0.0,
            "ram_used_mb": 0.0,
            "prompt_tokens": len(prompt.split()),
            "output_tokens": 10,
            "total_tokens": len(prompt.split()) + 10,
            "tokens_per_second": 100.0,
        },
        "full_text": f"TensorRT-LLM stub response for prompt: {prompt}",
    }
