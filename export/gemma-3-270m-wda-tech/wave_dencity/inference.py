import torch
import torch.nn.functional as F
from contextlib import nullcontext

@torch.inference_mode()
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temp: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed_text: str = "",
    device: str = "cuda",
):
    """Generate text from a prompt using BPE tokenizer."""
    model.eval()
    seq_len = model.seq_len

    dev_str = str(device)
    if dev_str.startswith("cuda"):
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif dev_str == "mps":
        autocast_ctx = lambda: torch.autocast(device_type="mps", dtype=torch.float16)
    else:
        autocast_ctx = None
    
    seed_tokens = tokenizer.encode(seed_text, add_special_tokens=False) if seed_text else []
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = seed_tokens + prompt_tokens
    
    if len(tokens) >= seq_len:
        idx = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
    else:
        try:
            pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        except Exception:
            pad_id = 0
        padding = [pad_id] * (seq_len - len(tokens))
        idx = torch.tensor([padding + tokens], dtype=torch.long, device=device)
    
    generated: list[int] = []
    with (autocast_ctx() if autocast_ctx is not None else nullcontext()):
        for _ in range(max_tokens):
            logits = model(idx)
            next_logits = logits[:, -1, :]

        if repetition_penalty != 1.0 and generated:
            prev = torch.tensor(generated, device=next_logits.device, dtype=torch.long)
            next_logits[0, prev] = next_logits[0, prev] / float(repetition_penalty)

            if temp and temp > 0:
                next_logits = next_logits / float(max(temp, 1e-6))

        if top_k and top_k > 0:
            v, _ = torch.topk(next_logits, k=min(int(top_k), next_logits.shape[-1]))
            cutoff = v[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)

        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumprobs > float(top_p)
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            next_logits = torch.full_like(next_logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            if temp and temp <= 0:
                nxt = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx[:, 1:], nxt], dim=1)
            generated.append(int(nxt.item()))
    
    return tokenizer.decode(generated)
