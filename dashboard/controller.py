import io
import torch
import requests
from concurrent.futures import ThreadPoolExecutor

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class Controller:
    def __init__(self, model_id=""):
        self.model_id = model_id
        self.weights_path = f"finetuned-{self.model_id}"
        # self.nodes = ["http://localhost:5001", "http://localhost:5002"]
        self.nodes = ["http://0.0.0.0:5001"]
        self.metrics = { "loss": 0.0, "step": 0 }

        self._initialize_model()

    def _initialize_model(self):
        self.model = TinyModel()
        
    def _save_weights(self):
        torch.save(self.model, self.weights_path)

    def _send_weights(self, url):
        # requests.get(url)
        with open(self.weights_path, "rb") as w:
            weights={ "weights": w }
            steps={ "step": self.metrics["step"] }
            requests.post(f"{url}/train", files=weights, data=steps)

    def initialize_node_weights(self):
        self._save_weights()
        print(self.nodes)
        with ThreadPoolExecutor() as executor:
            executor.map(lambda n: self._send_weights(n), self.nodes)

    def get_nodes(self):
        gradients = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(requests.get, f"{n}/gradients") for n in self.nodes]
            
            for future in futures:
                try:
                    response = future.result(timeout=5)  # Add timeout to avoid hanging
                    if response.status_code == 200:
                        gradient = torch.load(io.BytesIO(response.content), weights_only=False)
                        gradients.append(gradient)
                    else:
                        print(f"Failed to get gradients: HTTP {response.status_code}")
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error: {e}")
                except Exception as e:
                    print(f"Error retrieving gradients: {e}")
                    
        return gradients
