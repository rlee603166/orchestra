from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import threading
import subprocess
import torch
import json


"""Hyperparameters"""
MODEL_NAME = "EleutherAI/pythia-6.9b"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LENGTH = 128
MAX_ITERS = 50
LOG_FILE = "training_data.json"


"""App Setup"""
app = FastAPI()
asgi_app = WsgiToAsgi(app)


""""Model and tokenizer"""
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(DEVICE)

"""Data"""
dummy_text = ["Hello world" * 10] * BATCH_SIZE
batch = tokenizer(
    dummy_text,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    truncation=True,
    padding=True
)

""""Trainining"""
stop_training = threading.Event()
log_data = { "steps": [], "loss": [], "accuracy":[] }
current_step = 0
last_loss = 0.0
last_accuracy = 0.0


def training_loop():
    global current_step, last_loss
    while not stop_training.is_set() and current_step < MAX_ITERS:
        optimizer.zero_grad()
        with torch.amp.autocast(DEVICE):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        labels = batch["input_ids"]
        mask = batch["attention_mask"].bool()
        correct = (preds == labels) & mask
        accuracy = correct.sum().float() / mask.sum()

        scaler.scaler(loss).backwards()
        scaler.step(optimizer)
        scaler.update()

        last_loss = loss.item()
        last_accuracy = accuracy.item()
        log_data["steps"].append(current_step)
        log_data["loss"].append(last_loss)
        log_data["accuracy"].append(last_accuracy)
        with open(LOG_FILE, "w") as f:
            json.dump(log_data, f)
        
        current_step+=1
        print(f"Step: {current_step}/{MAX_ITERS}, Loss: {last_loss}, Accuracy: {last_accuracy}")


def get_specific(name):
    model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


@app.get("/")
def hello():
    return { "message": "Hello from compute node"}

@app.post("/pretrained-inference")
def pretrained_inference(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_length=100)
    return { "response": tokenizer.decode(outputs[0], skip_special_tokens=True) }


@app.get("/train")
def train():
    global stop_training, current_step
    stop_training.clear()
    current_step = 0
    threading.Thread(target=training_loop, daemon=True).start()       
    return { "status": "success" }
   

@app.post("/stop")
def stop():
    global stop_training
    stop_training.set()
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
