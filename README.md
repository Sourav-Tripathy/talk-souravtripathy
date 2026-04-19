# talk.souravtripathy.com

This project is a personal endeavor to keep my old NVIDIA GeForce GTX 1650 Ti busy, since it sits idle most of the time in my laptop. It is an attempt to host a 500M parameter Qwen model via vLLM, wrapped in a FastAPI server.

## Current Status
The backend **works**, but is currently suffering from bloat or inefficiency. This occasionally makes the laptop freeze and sometimes throws Out-Of-Memory (OOM) errors during concurrent calls. Given the specifications of a 1650 Ti with a 500M model, it should easily handle 4-5 concurrent calls without OOMing. Active optimizations and bug fixes are planned.

Once stabilized and fixed, the project will be hosted live at [talk.souravtripathy.com](https://talk.souravtripathy.com).

## vLLM Configuration
Because the GTX 1650 Ti has a Turing generation architecture (Compute Capability 7.5), it lacks support for certain modern optimized kernels like FlashAttention 2 or FlashInfer. The vLLM setup has been configured to work around these limitations:
- **Attention Backend:** Enforced `TRITON_ATTN`. This prevents vLLM from attempting FlashInfer JIT compilation, which would otherwise require `nvcc` and cause crash failures during engine initialization.
- **Enforce Eager:** `enforce_eager=True` is enabled to skip CUDA graph capture. This avoids the heavy VRAM allocation overhead during the profiling loop, which speeds up startup time and prevents instant OOMs on low-VRAM cards like the 4GB 1650 Ti.

## Credits 
- **[Qwen](https://github.com/QwenLM/Qwen)** - For their highly capable set of open-weight models, making the 500M model perform extraordinarily well on a local device.
- **[vLLM](https://github.com/vllm-project/vllm)** - For an incredibly fast and efficient LLM serving framework.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
