# import subprocess
# import sys

# pip_command = (
#     f'CMAKE_ARGS=-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" {sys.executable} -m pip install llama-cpp-python'
# )
# subprocess.check_call(pip_command, shell=True)

from llama_cpp import Llama

llm = Llama(
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\phi-4-GGUF\\phi-4-Q6_K.gguf",                                                     # 14B model,   Q6_K     - 20-25 sec for each response, model loading 50sec   (not generating the responses properly) 
    model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",       # 8B model,    Q8_0     - 35-45 sec for each response, model loading 9sec
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf",       # 8B model,    Q6_K
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Qwen-7B-GGUF\\DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf",         # 7B model,    Q6_K     - 161-165 sec for each response, model loading 25sec
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\hugging-quants\\Llama-3.2-1B-Instruct-Q8_0-GGUF\\llama-3.2-1b-instruct-q8_0.gguf",                    # 1B model,    q8_0
    n_gpu_layers=-1,  # Enables GPU acceleration
    verbose=True,
    logits_all=True,
    chat_format="llama-2",
    n_ctx=2000, 
    # seed=1337,         # Optional: uncomment to set a specific seed
    # n_ctx=2048,        # Optional: uncomment to increase the context window
)

print(llm)

# Include a system prompt to provide context, along with the user prompt.
messages = [
    {"role": "system", "content": "You are an assistant who provides helpful cooking advice."},
    {"role": "user", "content": "how to cook rice?use emojis"}
]

# Create a chat completion with streaming enabled
response = llm.create_chat_completion(
    messages=messages,
    temperature=0.2,
    top_p=0.9,         # Corrected top_p value
    max_tokens=1000,
    stream=False,
    
)

print("\n\n\n\n---------------------------output---------------------------")
print(type(response))
print(response)
print(print(response['choices'][0]['message']['content']))