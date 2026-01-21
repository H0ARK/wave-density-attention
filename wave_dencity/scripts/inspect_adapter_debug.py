import torch
from wave_dencity.transplant.adapter import load_adapter
from transformers import AutoModelForCausalLM
import json

def inspect_mismatch():
    path = "adapter_step1000.pt"
    # Try loading directly
    data = torch.load(path, map_location="cpu")
    v_keys = list(data["adapter"].keys())
    print(f"Total keys in adapter: {len(v_keys)}")
    
    # Check a few sample shapes
    for k in v_keys[:5]:
        print(f"  {k}: {data['adapter'][k].shape}")
        
    # Check if we can find a layer-specific param
    sample_key = [k for k in v_keys if "0.wda_attn.waves" in k]
    if sample_key:
        print(f"Sample waves shape: {data['adapter'][sample_key[0]].shape}")

if __name__ == "__main__":
    inspect_mismatch()
