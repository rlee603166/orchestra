from asgiref.wsgi import WsgiToAsgi
from flask import Flask, jsonify
from model import Model, get_batch
from flask_cors import CORS
import threading
import subprocess
import torch
import json

device = "cuda"
MAX_ITERS = 5000
LOG_FILE = "training_data.json"
log_data = { "steps": [], "loss": [], "accuracy":[] }

stop_training = threading.Event()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
asgi_app = WsgiToAsgi(app)


def get_gpu_usage():
    try:
        return subprocess.check_output(["rocm-smi", "--showuse"], universal_newlines=True)
    except FileNotFoundError:
        return "ROCm not installed or `rocm-smi` not found"    

def training_loop():
    while not stop_training.is_set():
        model = Model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for iter in range(MAX_ITERS):
            xb, yb = get_batch('train')

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            log_data["steps"].append(iter)
            log_data["loss"].append(loss.item())
            # log_data["accuracy"].append(logits.item())
            with open(LOG_FILE, "w") as f:
                    json.dump(log_data, f)


@app.route("/")
def hello():
    return { "message": "Hello from compute node"}


@app.route("/train", methods=["POST"])
def train():
    try:
        global stop_training
        stop_training.clear()
        threading.Thread(target=training_loop, daemon=True).start()       
        return jsonify({ "status": "success" })
    except:
        return jsonify({ "status": "failed" })


@app.route("/stats")
def stats():
    # nvidia gpus
    # gpu_usage = torch.cuda.utilization()
    # return jsonify({ "gpu": gpu_usage }) 
    gpu_usage = get_gpu_usage()
    return jsonify({ "gpu": gpu_usage })    


@app.route("/training_data")
def training_data():
    with open(LOG_FILE, "r") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)
