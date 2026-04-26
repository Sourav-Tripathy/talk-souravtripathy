import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import uuid
import torch
import time
import config

def print_memory_stats(label: str, func_name: str):
    print("\n" + "=" * 60)
    print(f"[{func_name}] -> {label}")
    print("-" * 60)
    
    # CPU Memory
    cpu_mem_mb = 0.0
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    cpu_mem_mb = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    print(f"[CPU] Current RSS Memory: {cpu_mem_mb:.2f} MB")
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu_mem_all_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_mem_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"[GPU] Memory Allocated:   {gpu_mem_all_mb:.2f} MB")
        print(f"[GPU] Memory Reserved:    {gpu_mem_res_mb:.2f} MB")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"[GPU] Free Memory:        {free_mem / (1024*1024):.2f} MB")
            print(f"[GPU] Total VRAM:         {total_mem / (1024*1024):.2f} MB")
        except Exception:
            pass
    else:
        print("[GPU] CUDA not available")
    print("=" * 60 + "\n")


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

# Engine is initialised explicitly via init_engine(). it is because we are intilaizing CUDA and pytorch says to use spawn and when using spawn if we donot call after parent process start rather intialize directly while fastapi server is starting, it will crash because of CUDA initialization in the parent process. By calling init_engine() after the server starts, we ensure that CUDA is initialized in the child process where it is needed, avoiding the crash and allowing the engine to function properly. This approach also allows us to control when the engine is initialized, which can be beneficial for resource management and debugging.
engine = None

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

def get_engine():
    if engine is None:
        raise RuntimeError("Engine not initialized")
    return engine


async def generate_stream(prompt: str):
    """
    Run async generation via vLLM and stream tokens back.
    Yields JSON chunks containing either tokens or final metrics.
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
    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    print_memory_stats("START of text generation", "generate_stream")

    full_output = ""
    last_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        last_output = output
        if output.outputs:
            new_text = output.outputs[0].text[len(full_output):]
            if new_text:
                yield {"type": "token", "content": new_text}
            full_output = output.outputs[0].text

    elapsed = time.perf_counter() - t0
    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    print_memory_stats("END of text generation", "generate_stream")

    prompt_tokens = len(last_output.prompt_token_ids) if last_output else len(prompt.split())
    output_tokens = len(last_output.outputs[0].token_ids) if last_output and last_output.outputs else len(full_output.split())
    total_tokens = prompt_tokens + output_tokens

    metrics = {
        "latency_ms": round(elapsed * 1000, 2),
        "vram_delta_mb": round((vram_after - vram_before) / 1e6, 2),
        "vram_used_mb": round(vram_after / 1e6, 2),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(output_tokens / elapsed, 2) if elapsed > 0 else 0,
    }

    yield {"type": "metrics", "content": metrics, "full_text": full_output.strip()}
