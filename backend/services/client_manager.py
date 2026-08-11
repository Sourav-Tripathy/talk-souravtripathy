import config

# Dynamically route client functions based on config.ENGINE_TYPE
if config.ENGINE_TYPE == "sglang":
    from services.sglang_client import (
        init_engine,
        generate_stream,
        IS_CPU,
    )
else:
    from services.vllm_client import (
        init_engine,
        generate_stream,
        IS_CPU,
    )

__all__ = ["init_engine", "generate_stream", "IS_CPU"]
