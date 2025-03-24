import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from torch.cuda.amp import autocast, GradScaler
from .custom_callback import CustomCallback
from torch.utils.data import DataLoader
from datasets import load_dataset
from dotenv import load_dotenv
from torch.optim import AdamW
from trl import SFTTrainer
import subprocess
import threading
import torch
import json
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

class SFTService:
    def __init__(
        self,
        model_id, 
        device=None, 
        log_file="training_data.json", 
        max_iters=50, 
        prompt_style="", 
        train_prompt_style="",
        rank=0,
        received_weights="r1-finetuned.pth"
    ):
        """Hyperparams"""
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        self.max_iters = max_iters
        self.prompt_style = prompt_style
        self.train_prompt_style = train_prompt_style

        self.rank = rank
        self.weights_path = f"compute-node-{self.rank}.pth"
        self.received_weights = received_weights

        """Training State"""
        self.current_metrics = {
            "step": 0,
            "accuracy": 0.0,
            "loss": 0.0
        }
        """"Models and Data"""
        self.scaler = torch.amp.GradScaler()
        self._initialize_model()
        self._initialize_data()


    def _initialize_model(self):
        import os
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=hf_token,
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

    def _initialize_data(self):
        shard_size = 1000
        start_idx = (self.rank * 1000)
        end_idx = (start_idx + 1) + shard_size
        train_dataset = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT",
            "en",
            split=f"train[{start_idx}:{end_idx}]",
            trust_remote_code=True
        )
        self.train_dataset = train_dataset.map(self._format_prompts, batched=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
        self.train_iter = iter(self.train_loader)

    def _format_prompts(self, examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = self.train_prompt_style.format(input, cot, output) + self.EOS_TOKEN
            texts.append(text)

        return {
            "text": texts,
        }    

    # def _training_loop(self, step, batch):
    #     torch.autograd.set_detect_anomaly(True)
    #     inputs = self.tokenizer(
    #         batch["text"],
    #         return_tensors="pt",
    #         truncation=True, 
    #         max_length=512
    #     ).to(self.device)
    #     input_ids = inputs["input_ids"].clone()
    #     attention_mask = inputs["attention_mask"].clone()
    #     labels = input_ids.clone()        
    #     self.model.zero_grad(set_to_none=True)
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         use_cache=False
    #     )
        
    #     loss = outputs.loss
    #     loss.backward()
        
    #     with torch.no_grad():
    #         logits = outputs.logits.detach()
    #         predictions = torch.argmax(logits, dim=-1)
    #         mask = labels != -100
    #         correct = (predictions == labels) & mask
    #         accuracy = correct.sum().item() / mask.sum().item()
        
    #     self.current_metrics["step"] = step
    #     self.current_metrics["loss"] = loss.item()
    #     self.current_metrics["accuracy"] = accuracy

    def _training_loop(self, step, batch):
        batch = self._get_batch()
        inputs = self.tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=512
        ).to(self.device)
        labels = inputs["input_ids"].clone()
        self.optimizer.zero_grad()

        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                use_cache=False
            )
            loss = outputs.loss
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        with torch.no_grad():
            logits = outputs.logits.detach()
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (predictions == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
        self.current_metrics["step"] = step
        self.current_metrics["loss"] = loss.item()
        self.current_metrics["accuracy"] = accuracy
         
    def _get_batch(self):
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return batch

    def _load_received_weights(self):
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

    def _save_and_clean_weights(self):
        torch.save({n: p.grad.clone() for n, p in self.model.parameters()}, self.weights_path)
        os.remove(self.received_weights)

    def train(self, step):
        # self._load_received_weights()
        batch = self._get_batch()
        self._training_loop(step, batch)
        # self._save_and_clean_weights()
        return self.current_metrics

    def stop(self):
        """Stop training"""
        self.stop_training.set()
        print("Stop training requested - waiting for training loop to terminate...")
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
            
        return { 
            "status": "stopped", 
            "message": "Training stop signal sent and processed" 
        }

    def _get_gpu_temp(self):
        try:
            result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu', 
                    '--format=csv,noheader,nounits'
                ], 
                capture_output=True,
                text=True, 
                check=True
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError:
            return None

    def _get_gpu_util(self):
        try:
            result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu', 
                    '--format=csv,noheader,nounits'
                ], 
                capture_output=True,
                text=True, 
                check=True
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError:
            return None

    def _catogorize_gpu_status(self, util, mem_percent, temp):
        if util < 5 and mem_percent < 10 and temp < 50:
            return "idle"
        elif (5 <= util <= 80) and (10 < mem_percent <= 80) and (50<= temp <= 75):
            return "activate"
        elif (80 < util <= 95) or (80 < mem_percent <= 95) or (75 < temp <= 85):
            return "warning"
        elif util > 95 or mem_percent > 95 or temp > 85:
            return "critical"
        else:
            return "offline"

    def get_gpu_stats(self):
        if self.device == "cuda":
            used_memory = torch.cuda.memory_allocated(0) / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            util = self._get_gpu_util() 
            temp = self._get_gpu_temp() 
            status = self._catogorize_gpu_status(util, int(used_memory/total_memory*100), temp)
            return {
                "gpu_stats": {
                    "vram_used_gpu": round(used_memory, 2),
                    "vram_total_gb": round(total_memory, 2)
                },
                "gpu_util": util,
                "gpu_temp": temp,
                "status": status,
            }

    def get_training_data(self):
        with open(self.log_file, "r") as f:
            return json.load(f)
 
    def get_current_training_state(self):
        return self.callback_handler.get_current_stats()

    def run_inference(self, prompt):
        """Run inference with the model"""
        FastLanguageModel.for_inference(self.model)
        inputs = self.tokenizer([self.prompt_style.format(prompt, "")], return_tensors="pt").to(self.device)
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs)
        FastLanguageModel.for_training(self.model)
        return response[0].split("### Response:")[1]

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
