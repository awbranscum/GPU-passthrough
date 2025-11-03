# ollama_gpu_server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio
import time
import os

# ======= GPU Selection & Prioritization =======

# Load user-selected GPU priority from file (created by gpu_selector.py)
gpu_file = "selected_gpus.txt"
if os.path.exists(gpu_file):
    with open(gpu_file, "r") as f:
        gpu_list = f.read().strip()
        if gpu_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
            print(f"✅ Using selected GPUs (priority order): {gpu_list}")
else:
    print("⚠️ No GPU selection found. Run 'python gpu_selector.py' first.")

# ======= FastAPI App =======

app = FastAPI(title="Ollama-Compatible GPU Server")

# ======= Model Cache =======
model_cache = {}

def load_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]

    # Automatically select the first available GPU based on priority
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = "cpu"
        for i in range(num_gpus):
            try:
                # Try to allocate memory to test if device is usable
                torch.cuda.mem_get_info(i)
                device = f"cuda:{i}"
                print(f"✅ Using device {device} for model {model_name}")
                break
            except RuntimeError:
                continue
    else:
        device = "cpu"
        print("⚠️ No CUDA devices available. Running on CPU.")

    print(f"Loading model {model_name} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model_cache[model_name] = (tokenizer, model, device)
    return model_cache[model_name]

# ======= Ollama API Schemas =======
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: dict = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    options: dict = {}

# ======= Ollama API Endpoints =======

@app.get("/")
def root():
    return {"status": "GPU Ollama-compatible server running"}

@app.get("/api/tags")
def list_models():
    return {"models": [{"name": name} for name in model_cache.keys()]}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    tokenizer, model, device = load_model(req.model)
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=req.options.get("num_predict", 128))
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": result,
        "done": True,
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    tokenizer, model, device = load_model(req.model)
    conversation = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
    inputs = tokenizer(conversation, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=req.options.get("num_predict", 128))
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": result},
        "done": True,
    }

@app.get("/api/show")
def show_model_info(model: str):
    if model in model_cache:
        return {
            "name": model,
            "device": model_cache[model][2],
            "params": sum(p.numel() for p in model_cache[model][1].parameters()),
        }
    return {"error": "model not loaded"}

@app.on_event("startup")
async def startup_event():
    print("✅ Ollama-compatible GPU server ready.")
    print("Visit http://localhost:11434 to test.")
