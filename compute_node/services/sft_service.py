from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from .custom_callback import CustomCallback
from datasets import load_dataset
from dotenv import load_dotenv
from trl import SFTTrainer
import subprocess
import threading
import torch
import json
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

class SFTService:
    def __init__(self, model_id, device=None, log_file="training_data.json", max_iters=50, prompt_style="", train_prompt_style=""):
        """Hyperparams"""
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        self.max_iters = max_iters
        self.prompt_style = prompt_style
        self.train_prompt_style = train_prompt_style

        """Training State"""
        self.stop_training = threading.Event()
        self.training_thread = None
        self.callback_handler = CustomCallback(log_file=log_file)

        """"Models and Data"""
        self._initialize_model()
        self._initialize_data()
        self._setup_trainer()


    def _initialize_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=hf_token,
        )
        self.EOS_TOKEN = self.tokenizer.eos_token
        self._get_peft_model()

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
        train_dataset = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT",
            "en",
            split="train[0:500]",
            trust_remote_code=True
        )
        
        validation_dataset = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT",
            "en",
            split="train[500:600]",
            trust_remote_code=True
        )
        
        self.train_dataset = train_dataset.map(self._format_prompts, batched=True)
        self.validation_dataset = validation_dataset.map(self._format_prompts, batched=True)

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

    def _setup_trainer(self):
        training_args = TrainingArguments(
            output_dir="./r1-fine-tuned",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs = 1, # warmup_ratio for full training runs!
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            save_steps=self.max_iters // 5,
            evaluation_strategy="steps",
            eval_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            dataset_num_proc=1,
            args=training_args,
            packing=True,
        )

        self.trainer.add_callback(self.callback_handler)

    def _training_loop(self):
        try:
            torch.cuda.empty_cache()
            self.trainer.train()
        except Exception as e:
            print(f"Training error: {e}")
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

        if new_loop:
            # Reset callback data
            self.callback_handler = CustomCallback(log_file=self.log_file)
            self.trainer.add_callback(self.callback_handler)

        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        return { "status": "success" }

    def stop(self):
        """Stop training"""
        self.stop_training.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
            
        return { "status": "stopped" }

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
            status = self._catogorize_gpu_status(util, int(used_memory/total_memory), temp)
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