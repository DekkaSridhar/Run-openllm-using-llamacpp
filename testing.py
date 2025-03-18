import time
from llama_cpp import Llama

# Measure time for loading the LLM
start_load_time = time.time()

llm = Llama(
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\phi-4-GGUF\\phi-4-Q6_K.gguf",                                                     # 14B model,   Q6_K     - 20-25 sec for each response, model loading 50sec   (not generating the responses properly) 
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",       # 8B model,    Q8_0     - 35-45 sec for each response, model loading 9sec
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf",       # 8B model,    Q6_K
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Qwen-7B-GGUF\\DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf",         # 7B model,    Q6_K     - 161-165 sec for each response, model loading 25sec
    # model_path="C:\\Users\\ctrls\\.lmstudio\\models\\hugging-quants\\Llama-3.2-1B-Instruct-Q8_0-GGUF\\llama-3.2-1b-instruct-q8_0.gguf",                    # 1B model,    q8_0
    
    model_path="D:\\llm models\\.lmstudio\\models\\lmstudio-community\\DeepSeek-R1-Distill-Llama-8B-GGUF\\DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",       # 8B model,    Q8_0     - 35-45 sec for each response, model loading 9sec
    n_gpu_layers=-1,  # Enables GPU acceleration
    verbose=True,
    chat_format="llama-2",
    n_ctx=1500,                  # Adjust context window
    # use_cuda=True,               # Ensure CUDA is used
    # use_mlock=True,              # Locks memory to improve stability
    # use_mmap=True,               # Memory-map model for faster loading
    # gpu_layers_graph=True        # 🚀 Enable CUDA Graphs
)

end_load_time = time.time()
load_duration = end_load_time - start_load_time
print(f"\n\nModel Loading Time: {load_duration:.2f} seconds\n")

# List of questions
questions = [
    {"role": "user", "content": "small recipy of how to cook rice?"},
    {"role": "user", "content": "simple code to use function constructor and class and object in python"},
    {"role": "user", "content": "small passage on hanuman with using better symbols and emojis"}
]

# System prompt
system_message = {"role": "system", "content": "You are an assistant who provides helpful responses."}

# Generate responses and measure time for each
response_times = []  # To store response times

for i, question in enumerate(questions, start=1):
    print(f"\n\nProcessing Question {i}...")
    start_response_time = time.time()

    response = llm.create_chat_completion(
        messages=[system_message, question],
        temperature=0.2,
        top_p=0.9,
        max_tokens=2000,
        stop=["[/INST]"], # Explicit stopping condition
        stream=False
    )

    end_response_time = time.time()
    response_duration = end_response_time - start_response_time
    response_times.append(response_duration)

    print("---------------------------Output---------------------------")
    print(response['choices'][0]['message']['content'])
    print(f"\nResponse Generation Time for Q{i}: {response_duration:.2f} seconds")

# Total time calculation
total_time = load_duration + sum(response_times)

print(f"\n\nTotal Time (Model Loading + All Responses): {total_time:.2f} seconds")