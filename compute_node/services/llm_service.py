from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import threading
import torch
import json


class LLMService:
    def __init__(self, model_id, device=None, log_file="training_data.json", max_iters=50):
        """Hyperparams"""
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        self.max_iters = max_iters

        """Training State"""
        self.stop_training = threading.Event()
        self.training_thread = None
        self.log_data = { "steps": [], "loss": [], "accuracy":[] }
        self.current_step = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0

        """"Models and Data"""
        self._initialize_model()
        self._initialize_data()


    def _initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scaler = torch.amp.GradScaler()

    def _initialize_data(self):
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        tokenized_dataset = dataset.map(
            self._preprocess_data,
            batched=True
        )
        self.batch = tokenized_dataset.map(
            self._format_data, 
            remove_columns=["text"]
        )
        self._initialize_model()

    def _preprocess_data(self, example):
        """Data formatting and tokenization"""
        return self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    def _format_data(self, example):
        example["input_ids"] = self.tokenizer.encode(
            example["text"], 
            truncation=True,
            max_length=512
        )
        example["labels"] = example["input_ids"].copy()
        return example

    def _save_training_data(self):
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f)

    def _training_loop(self):
        try:
            torch.cuda.empty_cache()
            while not self.stop_training.is_set() and self.current_step < self.max_iters:
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(**self.batch, labels=self.batch["input_ids"])
                    loss = outputs.loss
                    logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                labels = self.batch["input_ids"]
                mask = self.batch["attention_mask"].bool()
                correct = (preds == labels) & mask
                accuracy = correct.sum().float() / mask.sum()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.last_loss = loss.item()
                self.last_accuracy = accuracy.item()
                self.log_data["steps"].append(self.current_step)
                self.log_data["loss"].append(self.last_loss)
                self.log_data["accuracy"].append(self.last_accuracy)
                
                self._save_training_data()
                self.current_step += 1
                print(f"Step: {self.current_step}/{self.max_iters}, Loss: {self.last_loss}, Accuracy: {self.last_accuracy}")
        finally:
            torch.cuda.empty_cache()
            print("Training thread terminated and CUDA memory cleaned")

    def start_training(self, new_loop=True):
        """Start training on a seperate thread"""
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training.set()
            self.training_thread.join(timeout=5)
            torch.cuda.empty_cache()
        
        self.stop_training.clear()
        self.current_step = 0

        if new_loop:
            self.log_data = { "steps": [], "loss": [], "accuracy":[] }
            self._save_training_data()

        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        return { "status": "success" }           

    def stop(self):
        """Stop training"""
        self.stop_training.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
            
        return { "status": "stopped" }

    def get_gpu_stats(self):
        if self.device == "cuda":
            return {
                "gpu_usage": {
                    "vram_used_gpu": torch.cuda.memory_allocated(0) / 1024**3,
                    "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
                }
            }

    def get_training_data(self):
        with open(self.log_file, "r") as f:
            return json.load(f)
            
    def run_inference(self, prompt):
        """Run inference with the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, max_length=100)
        return { "response": self.tokenizer.decode(outputs[0], skip_special_tokens=True)}

    def cleanup_memory(self):
        """Aggressively clean up CUDA memory"""
        before_allocated = torch.cuda.memory_allocated(0) / 1024**3
        before_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        self.stop_training.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        print("Moving model to CPU to free GPU memory...")
        model_state = self.model.state_dict()
        del self.model
        torch.cuda.empty_cache()
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        
        after_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        return {
            "status": "memory cleanup completed",
            "memory_before": {
                "allocated_gb": before_allocated,
                "reserved_gb": before_reserved
            },
            "memory_after": {
                "allocated_gb": after_allocated,
                "reserved_gb": after_reserved
            },
            "memory_freed_gb": before_allocated - after_allocated
        }


    def cleanup_resources(self):
        """Clean up any resources before application exit"""
        print("Cleaning up resources...")
        
        self.stop_training.set()
        
        if self.training_thread and self.training_thread.is_alive():
            print("Waiting for training thread to terminate...")
            self.training_thread.join(timeout=5)
        
        torch.cuda.empty_cache()
        print("Resources cleaned up")
