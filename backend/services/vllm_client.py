import os

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import uuid
import torch
import time
import config

ENGINE_ARGS = AsyncEngineArgs(
    model=config.MODEL_PATH,
    max_model_len=config.MAX_MODEL_LEN,
    gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
    dtype=config.DTYPE,
    # GTX 1650 Ti (compute cap 7.5): FA2 unsupported; force Triton backend so
    # vLLM never tries FlashInfer JIT compilation (which requires nvcc).
    attention_backend="TRITON_ATTN",
    # Skip CUDA graph capture — avoids the profiling loop that triggers the
    # FlashInfer crash and reduces startup time on low-VRAM cards.
    enforce_eager=True,
)

# Engine is initialised once at import time (module-level singleton).
# vLLM handles paged attention, KV cache, and async batching internally.
engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)


async def generate_stream(prompt: str):
    """
    Run async generation via vLLM and stream tokens back.
    Yields JSON chunks containing either tokens or final metrics.
    """
    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        top_p=config.TOP_P,
    )

    request_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    full_output = ""
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            new_text = output.outputs[0].text[len(full_output):]
            if new_text:
                yield {"type": "token", "content": new_text}
            full_output = output.outputs[0].text

    elapsed = time.perf_counter() - t0
    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    metrics = {
        "latency_ms": round(elapsed * 1000, 2),
        "vram_delta_mb": round((vram_after - vram_before) / 1e6, 2),
        "vram_used_mb": round(vram_after / 1e6, 2),
        # Rough token estimate — tokenizer-accurate count can be added later.
        "prompt_tokens": len(prompt.split()),
    }

    yield {"type": "metrics", "content": metrics, "full_text": full_output.strip()}
