import torch
from wave_dencity import WaveCharLM
from transformers import GPT2TokenizerFast

# Device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Load model with seq_len=256 to match checkpoint
checkpoint = torch.load('wave_lm_bpe.pt', map_location=device)
model = WaveCharLM(
    vocab_size=len(tokenizer),
    seq_len=256,  # Match checkpoint
    embed_dim=1024,
    num_layers=4,
    num_heads=4,
    num_masks=16,
    num_waves_per_mask=8,
    topk_masks=8,
    attn_alpha=3.0,
).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate text
prompt = 'The future of AI is'
tokens = tokenizer.encode(prompt)
pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
padding = [pad_id] * (256 - len(tokens))
idx = torch.tensor([padding + tokens], dtype=torch.long, device=device)

generated = []
for _ in range(50):
    logits = model(idx)
    probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
    nxt = torch.multinomial(probs, 1)
    idx = torch.cat([idx[:, 1:], nxt], dim=1)  # Shift window
    generated.append(nxt.item())

output = tokenizer.decode(generated)
print(f'Prompt: {prompt}')
print(f'Generated: {output}')