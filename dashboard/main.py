from click.types import CompositeParamType
from flask import Flask, jsonify, render_template, jsonify, request
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
from controller import Controller, TinyModel
import threading
import requests
import torch
import torch.nn as nn


model_id = "toy-model"
controller = Controller(model_id)

app = Flask(__name__)
CORS(app)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/model")
def model():
    return controller.weights_path

@app.route("/ping")
def ping():
    url = "http://0.0.0.0:5000/training_data"
    x = requests.get(url)
    return x.json()

@app.route('/train', methods=["POST"])
def train():
    thread = threading.Thread(target=controller.train_loop)
    thread.start()
    return { "status": "success" }

@app.route('/training_data')
def data():
    return controller.get_training_data()

@app.route('/compare')
def compare():
    """Compare model weights with a node's weights to check for consistency."""
    local_path = controller.weights_path
    
    try:
        response = requests.get(f"{controller.nodes[0]}/gradients")
        if response.status_code != 200:
            return jsonify({"error": f"Failed to get weights from node: HTTP {response.status_code}"}), 500
        
        temp_path = "temp_node_weights.pth"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        are_equal, differences = controller.compare_models_from_files(local_path, temp_path)
        
        result = {
            "equal": are_equal,
            "differences": differences
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/gradients')
def gradients():
    gradients = controller.get_nodes()
    return { "is_same": len(gradients) }

asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8081)
