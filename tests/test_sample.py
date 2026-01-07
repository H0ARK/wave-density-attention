"""Quick test script to generate samples from checkpoint."""
import torch
import torch.nn.functional as F
from wave_dencity import WaveCharLM

# Load checkpoint
checkpoint_path = "wave_lm_c4.pt"
print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu")

# Get vocab
stoi = ckpt['stoi']
itos = ckpt['itos']
vocab_size = len(itos)

print(f"Step: {ckpt['step']}")
print(f"Best val loss: {ckpt.get('best_val_loss', 'N/A')}")
print(f"Vocab size: {vocab_size}")
print("="*60)

# Recreate model
model = WaveCharLM(
    vocab_size=vocab_size,
    seq_len=128,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    num_masks=192,
    num_waves_per_mask=48,
    topk_masks=8,
    attn_alpha=3.0,
)
model.load_state_dict(ckpt['model'])
model.eval()

# Generate from different prompts
prompts = [
    "The ",
    "In the ",
    "A new ",
    "When ",
    "This is ",
]

@torch.no_grad()
def generate(prompt_str, steps=500, temperature=0.8):
    # Encode prompt
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt_str[-128:]]], dtype=torch.long)
    if idx.size(1) < 128:
        # Pad to seq_len
        pad = torch.zeros((1, 128 - idx.size(1)), dtype=torch.long)
        idx = torch.cat([pad, idx], dim=1)
    
    result = []
    for _ in range(steps):
        logits = model(idx)[:, -1, :]  # [1, V]
        probs = F.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        result.append(itos[next_tok.item()])
        idx = torch.cat([idx[:, 1:], next_tok], dim=1)
    
    return prompt_str + "".join(result)

print("\nGenerating samples (temperature=0.8, 500 chars each):\n")
for i, prompt in enumerate(prompts, 1):
    print(f"\n--- Sample {i}: prompt='{prompt}' ---")
    sample = generate(prompt, steps=500, temperature=0.8)
    print(sample)
    print()

print("\n" + "="*60)
print("Generating longer sample (temperature=0.9, 1000 chars):\n")
long_sample = generate("The ", steps=1000, temperature=0.9)
print(long_sample)
