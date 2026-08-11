import os
import time
import torch
import config

# ── Device mode ───────────────────────────────────────────────────────────────
IS_CPU: bool = config.IS_CPU

# ── Engine lifecycle ──────────────────────────────────────────────────────────
engine = None

def init_engine():
    global engine
    if engine is not None:
        return

    from services.vllm_client import print_memory_stats
    print_memory_stats("BEFORE SGLang Engine Initialization", "init_engine")

    try:
        from sglang import Engine
        
        if IS_CPU:
            print(
                f"[SGLang Engine] CPU mode: model={config.MODEL_PATH}, "
                f"max_model_len={config.MAX_MODEL_LEN}"
            )
            # Propagate the SGLang CPU environment variable
            os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
            engine = Engine(
                model_path=config.MODEL_PATH,
                context_length=config.MAX_MODEL_LEN,
                tp_size=1,
            )
        else:
            print(
                f"[SGLang Engine] GPU mode: model={config.MODEL_PATH}, "
                f"max_model_len={config.MAX_MODEL_LEN}, "
                f"gpu_mem={config.GPU_MEMORY_UTILIZATION}"
            )
            # GPU settings tailored to low VRAM cards (e.g., GTX 1650 Ti)
            engine = Engine(
                model_path=config.MODEL_PATH,
                context_length=config.MAX_MODEL_LEN,
                mem_fraction_static=config.GPU_MEMORY_UTILIZATION,
                tp_size=1,
                disable_cuda_graph=True,  # Bypass CUDA graph capture to prevent low VRAM OOMs
            )
            
        print_memory_stats("AFTER SGLang Engine Initialization (SUCCESS)", "init_engine")
    except Exception as e:
        print_memory_stats("CRASH during SGLang Engine Initialization", "init_engine")
        print(f"Exception details: {e}")
        raise

def get_engine():
    if engine is None:
        raise RuntimeError("SGLang Engine not initialized. Call init_engine() first.")
    return engine

# ── Streaming inference ───────────────────────────────────────────────────────

async def generate_stream(prompt: str):
    """
    Run async generation via SGLang and stream tokens back.

    Yields dicts:
      • {"type": "token",   "content": <str>}
      • {"type": "metrics", "content": <dict>, "full_text": <str>}
    """
    if engine is None:
        raise RuntimeError("SGLang engine is not initialized. Call init_engine() first.")

    from services.vllm_client import _rss_mb, print_memory_stats

    # Define sampling parameters
    sampling_params = {
        "temperature": config.TEMPERATURE,
        "max_new_tokens": config.MAX_TOKENS,
        "top_p": config.TOP_P,
    }

    t0 = time.perf_counter()

    # Capture baseline stats
    vram_before = torch.cuda.memory_allocated() if not IS_CPU else 0
    ram_before  = _rss_mb() if IS_CPU else 0.0

    print_memory_stats("START of text generation (SGLang)", "generate_stream")

    full_output = ""
    step = 0
    prompt_tokens = 0
    output_tokens = 0

    # Request generation stream
    stream = await engine.async_generate(
        prompt=prompt,
        sampling_params=sampling_params,
        stream=True
    )

    async for chunk in stream:
        # A chunk is a dictionary containing the delta text and optional metadata
        new_text = chunk.get("text", "")
        if new_text:
            step += 1
            print(f"[SGLang Stream Step {step}] Streaming chunk: {repr(new_text)}")
            yield {"type": "token", "content": new_text}
            full_output += new_text

        # Retrieve token usage from meta_info if available in the chunk
        meta_info = chunk.get("meta_info", {})
        if meta_info:
            prompt_tokens = meta_info.get("prompt_tokens", prompt_tokens)
            output_tokens = meta_info.get("completion_tokens", output_tokens)

    elapsed = time.perf_counter() - t0

    vram_after = torch.cuda.memory_allocated() if not IS_CPU else 0
    ram_after  = _rss_mb() if IS_CPU else 0.0

    print_memory_stats("END of text generation (SGLang)", "generate_stream")

    # Fallbacks if meta_info token counts were not provided in the stream
    if prompt_tokens <= 0:
        prompt_tokens = len(prompt.split())
    if output_tokens <= 0:
        output_tokens = len(full_output.split())
    total_tokens = prompt_tokens + output_tokens

    if IS_CPU:
        metrics = {
            "device":            "cpu",
            "latency_ms":        round(elapsed * 1000, 2),
            "ram_delta_mb":      round(ram_after - ram_before, 2),
            "ram_used_mb":       round(ram_after, 2),
            "vram_delta_mb":     0,
            "vram_used_mb":      0,
            "prompt_tokens":     prompt_tokens,
            "output_tokens":     output_tokens,
            "total_tokens":      total_tokens,
            "tokens_per_second":  round(output_tokens / elapsed, 2) if elapsed > 0 else 0,
        }
    else:
        metrics = {
            "device":            "gpu",
            "latency_ms":        round(elapsed * 1000, 2),
            "vram_delta_mb":     round((vram_after - vram_before) / 1e6, 2),
            "vram_used_mb":      round(vram_after / 1e6, 2),
            "ram_delta_mb":      0,
            "ram_used_mb":       0,
            "prompt_tokens":     prompt_tokens,
            "output_tokens":     output_tokens,
            "total_tokens":      total_tokens,
            "tokens_per_second":  round(output_tokens / elapsed, 2) if elapsed > 0 else 0,
        }

    yield {"type": "metrics", "content": metrics, "full_text": full_output.strip()}
