## Llama.cpp
Llama.cpp is known for its portability and efficiency designed to run optimally on CPUs and GPUs without requiring specialized hardware. It is a lightweight framework which makes it ideal for technology teams launching LLMs on smaller devices and local On-Prem machines such as Edge use case scenarios.
Llama.cpp: Lightweight and Fast LLM Inference
Llama.cpp is an optimized C++ implementation for running Meta’s Llama models efficiently on CPUs and GPUs, even on low-end hardware. It enables users to run large language models (LLMs) locally without relying on cloud-based APIs.

Key Features:
✅ Lightweight & Portable – Works on Windows, macOS, Linux, and even mobile devices.
✅ Supports GGUF Models – Uses GGUF format for optimized performance.
✅ CPU & GPU Acceleration – Runs on CPUs efficiently; supports CUDA, Metal, and OpenCL for GPU acceleration.
✅ Quantization Support – Reduces model size while maintaining reasonable accuracy (4-bit, 5-bit, etc.).
✅ Multi-threading & AVX Optimizations – Uses AVX/FMA instructions for better CPU performance.
✅ CLI and API Access – Run models via command line or integrate into applications.

Use Cases:
🔹 Running LLMs locally for privacy and offline usage
🔹 Developing AI-powered chatbots and assistants
🔹 Performing text generation, summarization, code completion, etc.
🔹 Experimenting with quantized models for efficiency

💡 Ideal for developers and AI researchers who need a fast, self-hosted LLM runtime! 

## LM studio
LM Studio is a desktop application that allows users to download, run, and interact with local large language models (LLMs) on their own machines. It provides an easy-to-use interface for running models like Llama, Mistral, and other GGUF-based models without needing an internet connection.

Key Features of LM Studio:
✅ Run LLMs Locally – No cloud dependency, full privacy.
✅ Supports GGUF Models – Compatible with models like Llama 2, Mistral, and more.
✅ User-Friendly Interface – No complex setup, just download and run.
✅ GPU Acceleration – Uses CUDA (NVIDIA) or Metal (Apple) for better performance.
✅ Chat & API Mode – Interact with models via chat or serve them as an API.

https://www.youtube.com/watch?v=nATRPPZ5dGE   // video to setup nvdia cudas for windows
https://stackoverflow.com/questions/77267346/error-while-installing-python-package-llama-cpp-python   // pakages to be included in the visual studio installer

https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d           # llama.cpp using the nvdia gpus

https://llama-cpp-python.readthedocs.io/en/latest/                                                   # llama cpp documentation ***
https://huggingface.co/docs/api-inference/tasks/chat-completion                                       # create_chat_completion docs ***

https://freedium.cfd/https://medium.com/@gonchogo/how-to-install-llama-cpp-with-cuda-on-windows-05008d14f603   # llama-cpp using the gpu
https://github.com/langchain-ai/langchain/discussions/25342                                                    # llama-cpp using the gpu