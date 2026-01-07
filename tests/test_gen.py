"""Quick test of token-by-token generation."""
import torch
from transformers import GPT2TokenizerFast
from wave_dencity import WaveCharLM, generate_text
import sys

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "wave_lm_bpe-v5_best.pt"

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device: {device}")
print(f"Loading checkpoint: {checkpoint}")
ckpt = torch.load(checkpoint, map_location=device)

model_cfg = ckpt.get('model_cfg', {})
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

print("Creating model...")
model = WaveCharLM(vocab_size=len(tokenizer), seq_len=256, **model_cfg).to(device)

print("Loading weights...")
if '_orig_mod.tok_emb.weight' in ckpt['model']:
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
else:
    state_dict = ckpt['model']
model.load_state_dict(state_dict, strict=False)

print(f"\n{'='*80}")
print("Testing generation with token-by-token output")
print(f"{'='*80}")

# Try prompts that match the STEM training data
prompts = [
    "The derivative of x^2 is",
    "In mathematics, the Pythagorean theorem states that",
    "Consider a triangle with sides of length 3 and 4. The hypotenuse is",
]

for prompt in prompts:
    print(f"\n\nPrompt: {prompt}")
    
    result = generate_text(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=30,
        temp=0.7,
        top_p=0.95,
        device=device,
    )
    
    print(f"\nFull result: {result}")
