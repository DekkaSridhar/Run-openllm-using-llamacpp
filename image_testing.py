import base64
import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
chat_handler = Llava15ChatHandler(clip_model_path="D:\\llm models\\.lmstudio\\models\\mys\\ggml_llava-v1.5-7b\\mmproj-model-f16.gguf")
# chat_handler = Llava15ChatHandler(clip_model_path="D:\\llm models\\.lmstudio\\models\\lmstudio-community\\gemma-3-4b-it-GGUF\\mmproj-model-f16.gguf")


llm = Llama(
    model_path="D:\\llm models\\.lmstudio\\models\\mys\\ggml_llava-v1.5-7b\\ggml-model-q4_k.gguf",
    # model_path="D:\\llm models\\.lmstudio\\models\\lmstudio-community\\gemma-3-4b-it-GGUF\\gemma-3-4b-it-Q8_0.gguf",
    chat_handler=chat_handler,
    n_gpu_layers=-1,  # Enables GPU acceleration
    verbose=True,
    n_ctx=10000, # n_ctx should be increased to accommodate the image embedding
)

def image_to_base64_data_uri(file_path):
    # Get file extension
    _, ext = os.path.splitext(file_path)
    mime_type = f"image/{ext[1:].lower()}"  # Remove the dot and convert to lowercase
    
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
file_path = 'input/invoice.jpg'
data_uri = image_to_base64_data_uri(file_path)


messages = [
    {"role": "system", "content": "You are an assistant who perfectly describes images."},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri }},
            {"type" : "text", "text": "Describe this above image in detail please. explain every thing in the image in detaoied way"}
        ]
    }
]

response= llm.create_chat_completion(
    messages = messages,
    temperature=0.2,
    top_p=0.9,
    max_tokens=2000,
    stop=["[/INST]"], # Explicit stopping condition
    stream=False
)
print("result is:")
print(response)
# print(response['choices'][0]['message']['content'])


