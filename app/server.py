# app/server.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch
from huggingface_hub import HfFolder





app = FastAPI()

MODEL_ID = "meta-llama/Llama-3.1-8B"

# Tạo thư mục offload nếu chưa có
OFFLOAD_DIR = Path("model_offload").resolve()
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)


token = HfFolder.get_token()
# Load tokenizer & model (offload nếu GPU không đủ)

MODEL_PATH = Path("models/llama3.1-8b").resolve()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    offload_folder="model_offload"
)

# Dữ liệu đầu vào
class PromptInput(BaseModel):
    prompt: str

# Prompt chuẩn LLaMA 3.1
def build_prompt(user_prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    )

# Endpoint chat
@app.post("/chat")
async def chat_endpoint(input_data: PromptInput):
    prompt = build_prompt(input_data.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result.split("assistant")[-1].strip()}
