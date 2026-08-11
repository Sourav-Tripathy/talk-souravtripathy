import os
import uuid
import time

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from utils.memory import MemoryTracker


import config

# ── Device mode ───────────────────────────────────────────────────────────────
# Exported so routes can skip Mongo writes / adapt responses on CPU mode.
IS_CPU: bool = config.IS_CPU

# ── Engine args ───────────────────────────────────────────────────────────────

if not IS_CPU:
    # ── GPU path ──────────────────────────────────────────────────────────────
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
else:
    # ── CPU path ──────────────────────────────────────────────────────────────
    # Per vLLM CPU docs:
    #   • device = "cpu" selects the CPU backend.
    #   • dtype  = "bfloat16" — float16 is unstable on PyTorch CPU.
    #   • enforce_eager = True — no CUDA graphs on CPU.
    #   • VLLM_CPU_KVCACHE_SPACE is set in config.py and propagated to the
    #     environment there, before any vLLM import touches the C++ layer.

    print(
        f"[vLLM Engine] CPU mode: dtype={config.DTYPE}, "
        f"kvcache={config.CPU_KVCACHE_SPACE} GiB, "
        f"max_model_len={config.MAX_MODEL_LEN}"
    )

    ENGINE_ARGS = AsyncEngineArgs(
        model=config.MODEL_PATH,
        max_model_len=config.MAX_MODEL_LEN,
        dtype=config.DTYPE,          # bfloat16
        enforce_eager=True,
    )



# ── Engine lifecycle ──────────────────────────────────────────────────────────
# Initialised explicitly via init_engine() (called from startup_event in
# main.py) rather than at import time.  This avoids the CUDA-in-parent-process
# crash when uvicorn uses the "spawn" multiprocessing start method.

engine: AsyncLLMEngine | None = None


def init_engine():
    global engine
    if engine is not None:
        return
    
    try:
        with MemoryTracker("Engine Initialization", "init_engine"):
            engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
    except Exception as e:
        print(f"Exception details: {e}")
        raise



def get_engine() -> AsyncLLMEngine:
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    return engine


# ── Streaming inference ───────────────────────────────────────────────────────

async def generate_stream(prompt: str):
    if engine is None:
        raise RuntimeError("vLLM engine is not initialized. Call init_engine() first.")

    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        top_p=config.TOP_P,
    )

    request_id = str(uuid.uuid4())
    
    tracker = MemoryTracker("text generation", "generate_stream")
    
    with tracker:
        full_output = ""
        last_output = None
        last_flush_token_count = 0
        step = 0
        
        async for output in engine.generate(prompt, sampling_params, request_id):
            last_output = output
            if output.outputs:
                completion = output.outputs[0]
                token_ids = completion.token_ids
                latest_token_id = token_ids[-1] if token_ids else None
                new_text = completion.text[len(full_output):]
                
                step += 1
                if new_text:
                    yield {"type": "token", "content": new_text}
                    last_flush_token_count = len(token_ids)
                
                full_output = completion.text

        # Fetch completion metrics
        elapsed = time.perf_counter() - tracker.start_time
        vram_after = torch.cuda.memory_allocated() if not IS_CPU else 0
        ram_after  = tracker.get_rss_mb() if IS_CPU else 0.0

        prompt_tokens  = len(last_output.prompt_token_ids) if last_output else len(prompt.split())
        output_tokens  = (
            len(last_output.outputs[0].token_ids)
            if last_output and last_output.outputs
            else len(full_output.split())
        )
        total_tokens = prompt_tokens + output_tokens

        if IS_CPU:
            metrics = {
                "device":          "cpu",
                "latency_ms":      round(elapsed * 1000, 2),
                "ram_delta_mb":    round(ram_after - tracker.ram_before, 2),
                "ram_used_mb":     round(ram_after, 2),
                "vram_delta_mb":   0,
                "vram_used_mb":    0,
                "prompt_tokens":   prompt_tokens,
                "output_tokens":   output_tokens,
                "total_tokens":    total_tokens,
                "tokens_per_second": round(output_tokens / elapsed, 2) if elapsed > 0 else 0,
            }
        else:
            metrics = {
                "device":          "gpu",
                "latency_ms":      round(elapsed * 1000, 2),
                "vram_delta_mb":   round((vram_after - tracker.vram_before) / 1e6, 2),
                "vram_used_mb":    round(vram_after / 1e6, 2),
                "ram_delta_mb":    0,
                "ram_used_mb":     0,
                "prompt_tokens":   prompt_tokens,
                "output_tokens":   output_tokens,
                "total_tokens":    total_tokens,
                "tokens_per_second": round(output_tokens / elapsed, 2) if elapsed > 0 else 0,
            }

        yield {"type": "metrics", "content": metrics, "full_text": full_output.strip()}



# TODO: Work on a vLLM profiling function — vLLM upfronts total allotted GPU/CPU
#       memory and manages its own paged table, so nvidia-smi / /proc/self/status
#       show the same value before or after generation for the same session.
