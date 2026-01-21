import os
import torch
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import GPT2TokenizerFast
from wave_dencity import WaveCharLM, generate_text

def run_remote_inference(prompt="User: What is the benefit of wave-density attention?\nAssistant:"):
    repo_id = "H0ARK/wave-density-130m"
    print(f"🚀 Downloading model from Hugging Face: {repo_id}...")
    
    # 1. Download files from Hub
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    
    # 2. Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"✅ Config loaded: {config}")
    
    # 3. Initialize Model
    # Remove keys that shouldn't be passed to the constructor if any, 
    # but our config matches WaveCharLM args.
    model_args = {k: v for k, v in config.items() if k not in ["vocab_size", "seq_len"]}
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    model = WaveCharLM(
        vocab_size=config.get("vocab_size", 50257),
        seq_len=config.get("seq_len", 256),
        **model_args
    ).to(device)
    
    # 4. Load Weights (Safetensors)
    state_dict = load_file(weights_path)
    
    # Handle possible _orig_mod prefix and non-strict loading
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
            
    # Use strict=False to avoid issues with non-essential buffers (toeplitz_indices, etc.)
    # or slight differences in head bias configuration.
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("✅ Weights loaded successfully (non-strict mode).")
    
    # 5. Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # 6. Generate
    print(f"\n📝 Prompt: {prompt}")
    print("Generating...")
    
    output = generate_text(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=100, 
        device=device,
        temp=0.7,
        top_p=0.9
    )
    
    print(f"\n✨ Response:\n{output}")

if __name__ == "__main__":
    import sys
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else "User: Explain wave-density attention in one sentence.\nAssistant:"
    run_remote_inference(user_prompt)
