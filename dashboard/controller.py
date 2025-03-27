import io
import json
import time
import torch
import requests
from torch.optim import AdamW
from concurrent.futures import ThreadPoolExecutor
from unsloth import FastLanguageModel, is_bfloat16_supported

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
        self.weights_path = f"finetuned-DeepSeek-R1-Distill-Llama-8B.pth"
        # self.nodes = ["http://localhost:5001", "http://localhost:5002"]
        self.nodes = [
            "http://128.151.20.130:5000",
            "http://128.151.20.120:5000",
            # "http://128.151.20.147:5000",
            # "http://128.151.20.235:5000"
        ]
        self.metrics = { "step": [], "loss": [], "accuracy": [] }
        self.step = 0

        self.max_iters = max_iters
        self.training_log = training_log

        self._initialize_model()

    def _initialize_model(self):
        import os
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        self.EOS_TOKEN = self.tokenizer.eos_token
        self._get_peft_model()
        FastLanguageModel.for_training(self.model)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

    def _get_peft_model(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  
            bias="none",  
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,  
            loftq_config=None,
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        import os
        save_dir = os.path.join(os.path.expanduser("~"), "orchestra/dashboard/model_weights")
        os.makedirs(save_dir, exist_ok=True)
        
        self.weights_path = os.path.join(save_dir, "finetuned-DeepSeek-R1-Distill-Llama-8B.pth")
        
        model_save_dir = os.path.join(save_dir, "model")
        os.makedirs(model_save_dir, exist_ok=True)
        self.model.save_pretrained(model_save_dir)
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Model weights initialized and saved to {self.weights_path}")

    def _save_weights(self):
        import os
        save_dir = os.path.join(os.path.expanduser("~"), "model_weights")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Model weights saved to {self.weights_path}")

    def _send_weights(self, url):
        try:
            with open(self.weights_path, "rb") as w:
                files = {
                    "weights": (self.weights_path, w, "application/octet-stream")
                }
                data = {"step": str(self.step)}

                print(f"Sending weights to {url}")
                training_result = requests.post(f"{url}/train", files=files, data=data)
                print(f"Response status: {training_result.status_code}")
                return training_result
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error sending weights to {url}: {e}")
            return { "error": f"Connection error: {e}" }
        except Exception as e:
            print(f"Error sending weights to {url}: {e}")
            return { "error": f"Error in training: {e}" }

    def initialize_node_weights(self):
        with ThreadPoolExecutor() as executor:
            executor.map(lambda n: self._send_weights(n), self.nodes)

    def get_nodes(self):
        print(f"Attempting to get gradients from nodes: {self.nodes}")
        gradients_dict = {}
        nodes_responded = 0
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(requests.get, f"{n}/gradients") for n in self.nodes]
            
            for i, future in enumerate(futures):
                try:
                    print(f"Waiting for response from node {self.nodes[i]}")
                    response = future.result(timeout=15)  # Increased timeout
                    print(f"Received response from {self.nodes[i]}: {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"Content length: {len(response.content)} bytes")
                        
                        try:
                            node_gradients = torch.load(io.BytesIO(response.content), map_location="cpu")
                            print(f"Successfully loaded gradients from {self.nodes[i]}")
                            if not gradients_dict:
                                for name, grad in node_gradients:
                                    gradients_dict[name] = grad.clone()
                            else:
                                for name, grad in node_gradients:
                                    if name in gradients_dict:
                                        gradients_dict[name] += grad
                            
                            nodes_responded += 1
                        except Exception as e:
                            print(f"Error loading gradients from {self.nodes[i]}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"Failed to get gradients from {self.nodes[i]}: HTTP {response.status_code}")
                        if hasattr(response, 'text') and response.text:
                            print(f"Response content: {response.text[:200]}...")
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error to {self.nodes[i]}: {e}")
                except Exception as e:
                    print(f"Error retrieving gradients from {self.nodes[i]}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if nodes_responded == 0:
            print("No gradients were retrieved from any nodes")
            return {"status": "failed", "reason": "no gradients retrieved"}
        
        print(f"Successfully retrieved gradients from {nodes_responded} nodes")
        for name in gradients_dict:
            gradients_dict[name] /= nodes_responded
        
        gradient_names = set(gradients_dict.keys())
        print("Applying averaged gradients to the model...")
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        lr = 2e-5  # Default if not available from optimizer
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        
        total_params = 0
        updated_params = 0
        missing_gradients = []
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                if name in gradients_dict:
                    updated_params += 1
                    param_device = param.device
                    gradient = gradients_dict[name].to(param_device)
                    with torch.no_grad():
                        param.data.add_(gradient, alpha=-lr)
                else:
                    missing_gradients.append(name)
        
        print(f"Model has {total_params} parameters, updated {updated_params}")
        if missing_gradients:
            print(f"Missing gradients for {len(missing_gradients)} parameters")
            print(f"Some missing gradients: {missing_gradients[:5]}")
        
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Saved updated model weights to {self.weights_path}")
        
        return {"status": "averaged weights and updated model successfully"}

    def _train_nodes(self):
        print("Training nodes...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._send_weights, n) for n in self.nodes]
            node_loss = 0
            node_accuracy = 0
            nodes_responded = 0

            for future in futures:
                try:
                    response = future.result()  # Increased timeout
                    response_text = response.content.decode('utf-8')  
                    response_data = json.loads(response_text)
                    print(f"Training response: {response_data}")
                    
                    if response_data and "metrics" in response_data:
                        node_loss += response_data["metrics"]["loss"]
                        node_accuracy += response_data["metrics"]["accuracy"]
                        nodes_responded += 1
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error during training: {e}")
                except Exception as e:
                    print(f"Error in training: {e}")
                    import traceback
                    traceback.print_exc()
        
        if nodes_responded > 0:
            self.metrics["accuracy"].append(node_accuracy / nodes_responded)
            self.metrics["loss"].append(node_loss / nodes_responded)                
            self.metrics["step"].append(self.step)
            print(f"Updated metrics at step {self.step}: loss={node_loss/nodes_responded}, accuracy={node_accuracy/nodes_responded}")
        else:
            print("No nodes responded during training")

    def _update_stats(self):
        with open(self.training_log, "w") as f:
            json.dump(self.metrics, f)
        print(f"Updated statistics in {self.training_log}")

    def _reset_stats(self):
        with open(self.training_log, "w") as f:
            self.metrics = { "step": [], "loss": [], "accuracy": [] } 
            json.dump(self.metrics, f)
        print("Reset training statistics")

    def train_loop(self, stop_training):
        self._reset_stats()
        while self.step <= self.max_iters and not stop_training.is_set():
            print(f"Starting training step: {self.step}")
            self._train_nodes()
            print("Updating stats...")
            self._update_stats()
            
            print("Collecting and averaging gradients...")
            start = time.time()
            result = self.get_nodes()
            end = time.time()
            print(f"Gradient averaging completed in {end-start:.2f} seconds. Result: {result}")
            
            self.step += 1
            print(f"Completed step {self.step-1}")

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
        
        # Handle the case where we might be comparing gradients dictionaries
        if isinstance(model1, dict) and isinstance(model2, dict):
            keys1 = set(model1.keys())
            keys2 = set(model2.keys())
            
            if keys1 != keys2:
                if verbose:
                    print("Models have different parameter names:")
                    print(f"Only in model1: {keys1 - keys2}")
                    print(f"Only in model2: {keys2 - keys1}")
                return False
            
            all_equal = True
            differences = {}
            
            for key in keys1:
                tensor1 = model1[key]
                tensor2 = model2[key]
                
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
            
        # Handle traditional state_dict models
        state_dict1 = model1.state_dict() if hasattr(model1, 'state_dict') else model1
        state_dict2 = model2.state_dict() if hasattr(model2, 'state_dict') else model2
        
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