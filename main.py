import time
import random
from flask import Flask, jsonify, render_template, jsonify

app = Flask(__name__)

simulated_data = { "steps": [], "loss": [] }
start_time = time.time()

def simulate_training():
    elapsed = int(time.time() - start_time)
    steps = list(range(elapsed + 1))
    loss = []
    accuracy = []
    current_loss = 2.0
    current_acc = 0.2
    for _ in steps:
        current_loss = max(0.3, current_loss - random.uniform(0.01, 0.05))
        current_acc = min(0.95, current_acc + random.uniform(0.01, 0.03))
        loss.append(current_loss + random.uniform(-0.05, 0.05))
        accuracy.append(current_acc + random.uniform(-0.02, 0.02))
    return {"steps": steps, "loss": loss, "accuracy": accuracy}


@app.route('/training_data')
def training_data():
    return jsonify(simulate_training())


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)
