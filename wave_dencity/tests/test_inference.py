import torch
from wave_dencity.model_thought import WaveCharLM
from wave_dencity.inference import generate_text
from transformers import AutoTokenizer
import json

def test():
    config_path = "configs/scratch_nemotron_0.5b.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model with config
    model = WaveCharLM(
        dim=config.get("dim", 1024),
        n_layers=config.get("n_layers", 18),
        n_heads=config.get("n_heads", 16),
        seq_len=config.get("seq_len", 1024),
        vocab_size=config.get("vocab_size", 151936),
        thought_size=config.get("thought_size", 64)
    ).to(device)
    
    ckpt_path = "private/checkpoints/scratch_nemotron_wda/model_step300.pt"
    print(f"Loading {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Remove _orig_mod prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    prompt = "Hello, I am an AI trained from scratch."
    print(f"Prompt: {prompt}")
    output = generate_text(model, tokenizer, prompt, max_tokens=50, device=device)
    print(f"Output: {output}")

if __name__ == "__main__":
    test()
