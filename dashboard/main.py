from flask import Flask, jsonify, render_template, jsonify, request
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
from controller import Controller
import requests


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
    controller.initialize_node_weights()
    return { "status": "success" }

@app.route('/gradients')
def gradients():
    gradients = controller.get_nodes()
    return { "count": len(gradients) }

asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8081)
