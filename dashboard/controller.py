import io
import json
import time
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
    def __init__(self, model_id="", max_iters=100, training_log="training_log.json"):
        self.model_id = model_id
        self.weights_path = f"finetuned-{self.model_id}.pth"
        # self.nodes = ["http://localhost:5001", "http://localhost:5002"]
        self.nodes = [
            "http://128.151.20.130:5000",
            "http://128.151.20.120:5000",
            "http://128.151.20.147:5000",
            "http://128.151.20.235:5000"
        ]
        self.metrics = { "step": [], "loss": [], "accuracy": [] }
        self.step = 0

        self.max_iters = max_iters
        self.training_log = training_log

        self._initialize_model()

    def _initialize_model(self):
        self.model = TinyModel()
        self._save_weights()
        
    def _save_weights(self):
        torch.save(self.model, self.weights_path)

    def _send_weights(self, url):
        try:
            with open(self.weights_path, "rb") as w:
                files = {
                    "weights": (self.weights_path, w, "application/octet-stream")
                }
                data = {"step": str(self.step)}
                
                training_result = requests.post(f"{url}/train", files=files, data=data)
                return training_result
        except requests.exceptions.ConnectionError as e:
            return { "error": f"Connection error: {e}" }
        except Exception as e:
            return { "error": f"Error in training: {e}" }

    def initialize_node_weights(self):
        with ThreadPoolExecutor() as executor:
            executor.map(lambda n: self._send_weights(n), self.nodes)

    def get_nodes(self):
        gradients = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(requests.get, f"{n}/gradients") for n in self.nodes]
            
            for future in futures:
                try:
                    response = future.result(timeout=5)
                    if response.status_code == 200:
                        gradient = torch.load(io.BytesIO(response.content), weights_only=False)
                        gradients.append(gradient)
                    else:
                        print(f"Failed to get gradients: HTTP {response.status_code}")
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error: {e}")
                except Exception as e:
                    print(f"Error retrieving gradients: {e}")
        
        stacked = torch.stack(gradients)
        average_gradients = torch.mean(gradients, dim=0)
        torch.save(average_gradients, self.weights_path)
        return { "status": "averaged weights successfully" }

    def _train_nodes(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._send_weights, n) for n in self.nodes]
            node_loss = 0
            node_accuracy = 0

            for future in futures:
                try:
                    response = future.result(timeout=5)
                    print(response)
                    response = response.content.decode('utf-8')  
                    response = json.loads(response)
                    if response is not None:
                        node_loss+=response["metrics"]["loss"]
                        node_accuracy+=response["metrics"]["accuracy"]
                except requests.exceptions.ConnectionError as e:
                    return { "error": f"Connection error: {e}" }
                except Exception as e:
                    return { "error": f"Error in training: {e}" }

        self.metrics["accuracy"].append(node_accuracy / len(self.nodes))
        self.metrics["loss"].append(node_loss / len(self.nodes))                
        self.metrics["step"].append(self.step)

    def _update_stats(self):
        with open(self.training_log, "w") as f:
            json.dump(self.metrics, f)

    def _reset_stats(self):
        with open(self.training_log, "w") as f:
            self.metrics = { "step": [], "loss": [], "accuracy": [] } 
            json.dump(self.metrics, f)

    def train_loop(self, stop_training):
        self._reset_stats()
        while self.step <= self.max_iters and not stop_training.is_set():
            self._train_nodes()
            self._update_stats()
            start = time.time()
            self.get_nodes()
            end = time.time()
            print(f"gradient avg: {end-start}")
            self.step+=1

    def get_training_data(self):
        with open(self.training_log, "r") as f:
            return json.load(f)

    def compare_models_from_files(self, file1_path, file2_path, rtol=1e-5, atol=1e-8, verbose=True):
        try:
            model1 = torch.load(file1_path, weights_only=False)
            model2 = torch.load(file2_path, weights_only=False)
        except Exception as e:
            if verbose:
                print(f"Error loading models: {e}")
            return False
        
        if type(model1) != type(model2):
            if verbose:
                print(f"Models are of different types: {type(model1)} vs {type(model2)}")
            return False
        
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        
        if keys1 != keys2:
            if verbose:
                print("Models have different parameter names:")
                print(f"Only in model1: {keys1 - keys2}")
                print(f"Only in model2: {keys2 - keys1}")
            return False
        
        all_equal = True
        differences = {}
        
        for key in keys1:
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            
            if tensor1.shape != tensor2.shape:
                if verbose:
                    print(f"Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
                differences[key] = {"error": "shape_mismatch", "shapes": [list(tensor1.shape), list(tensor2.shape)]}
                all_equal = False
                continue
            
            is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
            
            if not is_close:
                abs_diff = (tensor1 - tensor2).abs()
                max_diff = abs_diff.max().item()
                max_diff_idx = torch.where(abs_diff == max_diff)
                if verbose:
                    print(f"Values differ for {key}. Max difference: {max_diff} at index {max_diff_idx}")
                differences[key] = {"error": "value_mismatch", "max_diff": max_diff}
                all_equal = False
        
        if all_equal and verbose:
            print("All model weights are equal within specified tolerance.")
        
        return all_equal, differences if not all_equal else None
