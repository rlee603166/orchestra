import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")

print("llama-3 loaded successfully!")