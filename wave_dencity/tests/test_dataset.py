import torch
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
ds_name = "nvidia/Llama-Nemotron-Post-Training-Dataset"
subset = None
split = "chat"

print(f"Testing load_dataset for {ds_name}...")
try:
    ds = load_dataset(ds_name, subset, split=split, streaming=True)
    item = next(iter(ds))
    print("Successfully loaded first item:")
    print(item)
except Exception as e:
    print(f"Error: {e}")
