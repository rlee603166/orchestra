from fastapi.responses import FileResponse
from prompts import prompt_style, train_prompt_style
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
# from services import SFTService
from typing import Annotated
import requests
import torch
import atexit
import signal
import sys
import os


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

# service = SFTService(
#     model_id=MODEL_NAME,
#     device=DEVICE,
#     max_iters=MAX_ITERS,
#     log_file=LOG_FILE,
#     prompt_style=prompt_style,
#     train_prompt_style=train_prompt_style
# )
#
# def cleanup_resources():
#     """Cleanup on exit"""
#     service.cleanup_resources()
#
# def handle_signal(signum, frame):
#     print(f"Received signal {signum}, exiting...")
#     cleanup_resources()
#     sys.exit(0)
#
#
# atexit.register(cleanup_resources)
# signal.signal(signal.SIGTERM, handle_signal)
# signal.signal(signal.SIGINT, handle_signal)
#

"""Ello"""
@app.get("/")
def hello():
    return { "message": "Hello from compute node" }


"""Training Endpoints"""
@app.post("/train")
async def train(
    weights: Annotated[UploadFile, File()],  # Binary weights file
    step: Annotated[int, Form()]
):
    print(f"step: {step}")
    with open(weights_path, "wb") as f:
        content = await weights.read()
        f.write(content)
    return { "yes": "yes"}
    # return service.start_training()

@app.get("/gradients")
def gradients():
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Gradients not found")
    return FileResponse(weights_path)

# @app.post("/stop")
# def stop():
#     return service.stop()
#
# @app.get("/status")
# def get_status():
#     data = service.callback_handler.current_stats
#     return { "status": data["is_training"] }
#
#
# @app.get("/training_data")
# def training_data():
#     return service.get_training_data()
#
#
# """GPU Endpoints"""
# @app.get("/stats")
# def stats():
#     return service.get_gpu_stats()
#
# @app.post("/cleanup_memory")
# def cleanup_memory():
#     return service.cleanup_memory()
#
#
# """Inference Endpoints"""
# class InferenceRequest(BaseModel):
#     model: str | None = None
#     prompt: str
#
# @app.post("/pretrained-inference")
# def pretrained_inference(request: InferenceRequest):
#     return service.run_inference(request.prompt)
#

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
