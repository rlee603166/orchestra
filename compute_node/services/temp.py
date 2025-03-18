from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import threading
import torch
import json


model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model = AutoModelForCausalLM.from_pretrained(model_id).to(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scaler = torch.amp.GradScaler()

def _preprocess_data(example):
    """Data formatting and tokenization"""
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

def _format_data( example):
    example["input_ids"] = tokenizer.encode(
        example["text"], 
        truncation=True,
        max_length=512
    )
    example["labels"] = example["input_ids"].copy()
    return example

dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenized_dataset = dataset.map(
    _preprocess_data,
    batched=True
)
formatted_dataset = tokenized_dataset.map(
    _format_data, 
    remove_columns=["text"]
)
print(formatted_dataset["train"][0])