from asgiref.wsgi import WsgiToAsgi
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Response, status
import threading
import subprocess
import torch
import json
import atexit
import signal
import sys


"""Hyperparameters"""
MODEL_NAME = "EleutherAI/pythia-6.9b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")
BATCH_SIZE = 1
MAX_LENGTH = 128
MAX_ITERS = 50
LOG_FILE = "training_data.json"


"""App Setup"""
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
asgi_app = WsgiToAsgi(app)


""""Model and tokenizer"""
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler()


"""Data"""
dummy_text = ["Hello world" * 10] * BATCH_SIZE
batch = tokenizer(
    dummy_text,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    truncation=True,
    padding=True
).to(DEVICE)


""""Training"""
stop_training = threading.Event()
training_thread = None
log_data = { "steps": [], "loss": [], "accuracy":[] }
with open(LOG_FILE, "w") as f:
    json.dump(log_data, f)

current_step = 0
last_loss = 0.0
last_accuracy = 0.0

def training_loop():
    global current_step, last_loss, last_accuracy
    try:
        torch.cuda.empty_cache()
        while not stop_training.is_set() and current_step < MAX_ITERS:
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
                logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            labels = batch["input_ids"]
            mask = batch["attention_mask"].bool()
            correct = (preds == labels) & mask
            accuracy = correct.sum().float() / mask.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            last_loss = loss.item()
            last_accuracy = accuracy.item()
            log_data["steps"].append(current_step)
            log_data["loss"].append(last_loss)
            log_data["accuracy"].append(last_accuracy)
            
            with open(LOG_FILE, "w") as f:
                json.dump(log_data, f)
            
            current_step += 1
            print(f"Step: {current_step}/{MAX_ITERS}, Loss: {last_loss}, Accuracy: {last_accuracy}")
    finally:
        torch.cuda.empty_cache()
        print("Training thread terminated and CUDA memory cleaned")


def cleanup_resources():
    """Clean up any resources before application exit"""
    global stop_training, training_thread
    print("Cleaning up resources...")
    
    stop_training.set()
    
    if training_thread and training_thread.is_alive():
        print("Waiting for training thread to terminate...")
        training_thread.join(timeout=5)
    
    torch.cuda.empty_cache()
    print("Resources cleaned up")


atexit.register(cleanup_resources)

def handle_signal(signum, frame):
    print(f"Received signal {signum}, exiting...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


@app.get("/")
def hello():
    return { "message": "Hello from compute node" }


"""Training Endpoints"""
@app.post("/train")
def train():
    global stop_training, current_step, log_data, training_thread
    
    if training_thread and training_thread.is_alive():
        stop_training.set()
        training_thread.join(timeout=5)
        torch.cuda.empty_cache()
    
    stop_training.clear()
    current_step = 0

    log_data = { "steps": [], "loss": [], "accuracy":[] }
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f)

    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()
    
    return Response(content='{ "status": "success" }', status_code=status.HTTP_200_OK)
   

@app.post("/stop")
def stop():
    global stop_training, training_thread
    stop_training.set()
    
    if training_thread and training_thread.is_alive():
        training_thread.join(timeout=5)
        
    return { "status": "stopped" }


@app.get("/stats")
def stats():
    return {
        "gpu_usage": {
            "vram_used_gpu": torch.cuda.memory_allocated(0) / 1024**3,
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    }


@app.get("/training_data")
def training_data():
    with open(LOG_FILE, "r") as f:
        return json.load(f)


@app.post("/cleanup_memory")
def cleanup_memory():
    """Aggressively clean up CUDA memory"""
    global model, DEVICE
    
    before_allocated = torch.cuda.memory_allocated(0) / 1024**3
    before_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    stop_training.set()
    if training_thread and training_thread.is_alive():
        training_thread.join(timeout=5)
    
    print("Moving model to CPU to free GPU memory...")
    model_state = model.state_dict()
    del model
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    
    after_allocated = torch.cuda.memory_allocated(0) / 1024**3
    after_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    return {
        "status": "memory cleanup completed",
        "memory_before": {
            "allocated_gb": before_allocated,
            "reserved_gb": before_reserved
        },
        "memory_after": {
            "allocated_gb": after_allocated,
            "reserved_gb": after_reserved
        },
        "memory_freed_gb": before_allocated - after_allocated
    }


"""Inference Endpoints"""
class InferenceRequest(BaseModel):
    model: str | None = None
    prompt: str

@app.post("/pretrained-inference")
def pretrained_inference(request: InferenceRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_length=100)
    return { "response": tokenizer.decode(outputs[0], skip_special_tokens=True) }