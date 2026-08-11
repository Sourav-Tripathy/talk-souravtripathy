import config

# Dynamically route client functions based on config.ENGINE_TYPE
if config.ENGINE_TYPE == "sglang":
    from services.sglang import (
        init_engine,
        generate_stream,
        IS_CPU,
    )
elif config.ENGINE_TYPE == "tensorrt":
    from services.tensorrt_llm import (
        init_engine,
        generate_stream,
        IS_CPU,
    )
elif config.ENGINE_TYPE == "transformer":
    from services.transformer import (
        init_engine,
        generate_stream,
        IS_CPU,
    )
else:
    # Default to vllm_client (vllm option)
    from services.vllm_client import (
        init_engine,
        generate_stream,
        IS_CPU,
    )

__all__ = ["init_engine", "generate_stream", "IS_CPU"]
