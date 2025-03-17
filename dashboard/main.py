from flask import Flask, jsonify, render_template, jsonify
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
import requests


app = Flask(__name__)
CORS(app)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/ping")
def ping():
    url = "http://0.0.0.0:5000/training_data"
    x = requests.get(url)
    return x.json()

@app.route('/train', methods=["POST"])
def train():
    url = "http://0.0.0.0:5000/train"
    x = requests.post(url)
    return x.json()


asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8081)
