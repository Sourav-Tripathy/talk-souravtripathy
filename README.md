# talk.souravtripathy.com

This is a learning project created to keep old hardware busy (such as my laptop's idle NVIDIA GeForce GTX 1650 Ti) while learning and exploring all the inference engines available. 

The project contains a lightweight static frontend and a backend exposing three endpoints:
- `POST /chat`: Streams model responses using Server-Sent Events (SSE).
- `GET /health`: Reports device and hardware system resource usage.
- `GET /logs/stream`: Streams live backend inference logs directly to the frontend.

## Supported Inference Engines
We support dynamic engine routing via the `ENGINE_TYPE` configuration. The engines are:
1. **vLLM** (`vllm_client.py`): Efficient serving engine configuration supporting both GPU and CPU pathways.
2. **SGLang** (`sglang.py`): Stub implementation for SGLang integration (to be completed).
3. **TensorRT-LLM** (`tensorrt_llm.py`): Stub implementation for NVIDIA TensorRT-LLM optimization (to be completed).
4. **Standard Transformer** (`transformer.py`): Stub implementation for a standard HuggingFace/PyTorch transformer runner (to be completed).

## Project Design
- **Standard logger:** Structured inference metrics are logged to `logs/inference.log` and streamed to the frontend.
- **vLLM low-VRAM optimizations:** Custom parameters like `attention_backend="TRITON_ATTN"` and `enforce_eager=True` are leveraged to run successfully on low-VRAM GPUs.

## License
This project is licensed under the MIT License.
