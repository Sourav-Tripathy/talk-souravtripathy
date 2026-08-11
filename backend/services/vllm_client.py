import os
import uuid
import time

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

import config

# ── Device mode ───────────────────────────────────────────────────────────────
# Exported so routes can skip Mongo writes / adapt responses on CPU mode.
IS_CPU: bool = config.IS_CPU


# ── Memory stats helper ───────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Return current process RSS in MB (reads /proc/self/status)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def print_memory_stats(label: str, func_name: str):
    print("\n" + "=" * 60)
    print(f"[{func_name}] -> {label}")
    print("-" * 60)

    # CPU / RAM
    print(f"[CPU] Current RSS Memory: {_rss_mb():.2f} MB")

    # GPU (only when CUDA is available)
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_res   = torch.cuda.memory_reserved()   / (1024 * 1024)
        print(f"[GPU] Memory Allocated:   {gpu_alloc:.2f} MB")
        print(f"[GPU] Memory Reserved:    {gpu_res:.2f} MB")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[GPU] Free Memory:        {free_mem / (1024*1024):.2f} MB")
            print(f"[GPU] Total VRAM:         {total_mem / (1024*1024):.2f} MB")
        except Exception:
            pass
    else:
        print("[GPU] CUDA not available — running in CPU mode")

    print("=" * 60 + "\n")


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

    print_memory_stats("BEFORE Engine Initialization", "init_engine")
    try:
        engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        print_memory_stats("AFTER Engine Initialization (SUCCESS)", "init_engine")
    except Exception as e:
        print_memory_stats("CRASH during Engine Initialization", "init_engine")
        print(f"Exception details: {e}")
        raise


def get_engine() -> AsyncLLMEngine:
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    return engine


# ── Streaming inference ───────────────────────────────────────────────────────

async def generate_stream(prompt: str):
    """
    Run async generation via vLLM and stream tokens back.

    Yields dicts:
      • {"type": "token",   "content": <str>}
      • {"type": "metrics", "content": <dict>, "full_text": <str>}

    On CPU mode:
      - VRAM fields are always 0 / N/A.
      - RAM delta (RSS) is reported instead.
    """
    if engine is None:
        raise RuntimeError("vLLM engine is not initialized. Call init_engine() first.")

    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        top_p=config.TOP_P,
    )

    request_id = str(uuid.uuid4())

    t0 = time.perf_counter()

    # Capture baseline stats
    vram_before = torch.cuda.memory_allocated() if not IS_CPU else 0
    ram_before  = _rss_mb() if IS_CPU else 0.0

    print_memory_stats("START of text generation", "generate_stream")

    full_output = ""
    last_output = None
    last_flush_token_count = 0
    step = 0
    async for output in engine.generate(prompt, sampling_params, request_id):
        last_output = output
        print("last_output: ", last_output)
        if output.outputs:
            completion = output.outputs[0]
            token_ids = completion.token_ids
            latest_token_id = token_ids[-1] if token_ids else None
            new_text = completion.text[len(full_output):]
            
            step += 1
            print(f"[vLLM Stream Step {step}] "
                  f"Token count: {len(token_ids)}, "
                  f"Latest Token ID: {latest_token_id}, "
                  f"Streaming chunk: {repr(new_text)}")
            print(f"vLLM completion object: {completion}")
            
            if new_text:
                yield {"type": "token", "content": new_text}
                len_diff = len(token_ids) - last_flush_token_count
                print(f"[vLLM Stream Step {step} After Flush] Diff of token id list length: {len_diff}")
                last_flush_token_count = len(token_ids)
            
            full_output = completion.text

    elapsed = time.perf_counter() - t0

    vram_after = torch.cuda.memory_allocated() if not IS_CPU else 0
    ram_after  = _rss_mb() if IS_CPU else 0.0

    print_memory_stats("END of text generation", "generate_stream")

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
            # RAM stats instead of VRAM
            "ram_delta_mb":    round(ram_after - ram_before, 2),
            "ram_used_mb":     round(ram_after, 2),
            # VRAM fields kept for API-schema compatibility — always 0 on CPU
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
            "vram_delta_mb":   round((vram_after - vram_before) / 1e6, 2),
            "vram_used_mb":    round(vram_after / 1e6, 2),
            # RAM field kept for API-schema compatibility — 0 on GPU path
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
