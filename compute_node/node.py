from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from prompts import prompt_style, train_prompt_style
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from llm_service import LLMService
from pydantic import BaseModel
from typing import Annotated
import requests
import signal
import atexit
import torch
import math
import sys
import os
import argparse


"""Parse command line arguments"""
parser = argparse.ArgumentParser(description='FastAPI compute node with rank support')
parser.add_argument('--rank', type=int, default=0, help='Node rank for distributed training')
parser.add_argument('--world-size', type=int, default=1, help='Total number of nodes')
parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
args = parser.parse_args()
print(f"Starting server with rank {args.rank}")


"""Hyperparameters"""
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")
BATCH_SIZE = 1
MAX_LENGTH = 128
MAX_ITERS = 50
LOG_FILE = "training_data.json"


weights_path = "weights.pth"
controller_url = "http://localhost:8081/train"

"""App Setup"""
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

service = LLMService(
    model_id=MODEL_NAME,
    device=DEVICE,
    max_iters=MAX_ITERS,
    log_file=LOG_FILE,
    prompt_style=prompt_style,
    train_prompt_style=train_prompt_style,
    rank=args.rank
)

def cleanup_resources():
    """Cleanup on exit"""
    service.cleanup_resources()

def handle_signal(signum, frame):
    print(f"Received signal {signum}, exiting...")
    cleanup_resources()
    sys.exit(0)



signal.signal(signal.SIGINT, handle_signal)


"""Ello"""
@app.get("/")
def hello():
    return { "message": "Hello from compute node" }


"""Training Endpoints"""
# @app.post("/train")
# async def train(
#     weights: Annotated[UploadFile, File()],
#     step: Annotated[int, Form()]
# ):
#     with open(service.received_weights, "wb") as f:
#         content = await weights.read()
#         f.write(content)
#     metrics = service.train(step) # Increasing accuracy function
#     return { "status": "train", "metrics": metrics }

# Artificial Data for demo
@app.post("/train")
async def train(
    # weights: Annotated[UplgccoadFile, File()],
    step: Annotated[int, Form()]
):
    metrics = { "step": step }
    initial_loss = 2.0
    final_loss = 0.1
    decay_rate = 0.01
    metrics["loss"] = final_loss + (initial_loss - final_loss) * math.exp(-decay_rate * step)

    initial_acc = 0.5
    final_acc = 0.98
    growth_rate = 0.015
    metrics["accuracy"] = final_acc - (final_acc - initial_acc) * math.exp(-growth_rate * step)
    
    return { "status": "train", "metrics": metrics }

@app.get("/gradients")
def gradients():
    print(f"Checking for weights at: {service.weights_path}")
    if not os.path.exists(service.weights_path):
        print(f"File not found: {service.weights_path}")
        raise HTTPException(status_code=404, detail="Gradients not found")
    file_size = os.path.getsize(service.weights_path)
    print(f"Found weights file, size: {file_size} bytes")
    return FileResponse(service.weights_path)

@app.post("/stop")
def stop():
    return service.stop()


"""GPU Endpoints"""
@app.get("/stats")
def stats():
    return service.get_gpu_stats()

@app.post("/cleanup_memory")
def cleanup_memory():
    return service.cleanup_memory()


"""Inference Endpoints"""
class InferenceRequest(BaseModel):
    model: str | None = None
    prompt: str

@app.post("/pretrained-inference")
def pretrained_inference(request: InferenceRequest):
    return service.run_inference(request.prompt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
