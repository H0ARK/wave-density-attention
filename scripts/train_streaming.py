"""
Wave-Density LM with Streaming Data
Trains on billions of tokens from C4 without downloading the full dataset.
"""
import math
import torch
import torch.nn.functional as F
from wave_dencity import WaveCharLM
from datasets import load_dataset
from typing import Iterator

def build_streaming_dataset(seq_len: int = 256, buffer_size: int = 10000):
    """Stream C4 data and convert to character indices on-the-fly."""
    # Load C4 with streaming enabled (no download)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Build character vocab from a sample (we'll use the same one from your checkpoint)
    print("Building vocab from sample...")
    sample_text = ""
    for i, example in enumerate(dataset):
        sample_text += example['text']
        if len(sample_text) > 1_000_000:  # 1M chars is enough
            break
    
    vocab = sorted(set(sample_text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"Vocab size: {len(vocab)}")
    
    # Character stream iterator
    def char_stream() -> Iterator[int]:
        """Infinite stream of character indices."""
        while True:
            shuffled = dataset.shuffle(buffer_size=buffer_size, seed=None)
            for example in shuffled:
                text = example['text']
                for char in text:
                    if char in stoi:
                        yield stoi[char]
                    elif char == '\n':
                        yield stoi.get('\n', stoi.get(' ', 0))
                    else:
                        yield stoi.get(' ', 0)
    
    return vocab, stoi, itos, char_stream()

def sample_batch(stream: Iterator[int], batch_size: int, seq_len: int, device: str):
    """Sample a batch from the character stream."""
    x_batch = []
    y_batch = []
    
    for _ in range(batch_size):
        # Get seq_len+1 characters
        chars = [next(stream) for _ in range(seq_len + 1)]
        x_batch.append(chars[:-1])
        y_batch.append(chars[1:])
    
    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x, y

def train_streaming(
    device: str = "cuda",
    seq_len: int = 256,
    steps: int = 50000,
    batch_size: int = 64,
    lr: float = 1e-3,
    checkpoint_path: str = "wave_lm_streaming.pt",
    save_every: int = 500,
    val_every: int = 100,
):
    """Train on streaming C4 data."""
    import time
    import os
    
    print("Initializing streaming dataset...")
    vocab, stoi, itos, char_stream = build_streaming_dataset(seq_len=seq_len)
    
    # Create model
    model = WaveCharLM(
        vocab_size=len(vocab),
        seq_len=seq_len,
        embed_dim=1024,
        num_layers=4,
        num_heads=4,
        num_masks=32,
        num_waves_per_mask=16,
        topk_masks=8,
        attn_alpha=3.0,
        content_rank=8,
        content_mix=0.15,
        learned_content=True,
    ).to(device)
    
    # Load checkpoint if exists
    start_step = 1
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0) + 1
        print(f"Resumed from step {start_step-1}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # LR scheduler with warmup and cosine decay
    warmup_steps = 500
    min_lr = lr * 0.1
    def get_lr(step):
        # 1) Linear warmup for warmup_steps
        if step < warmup_steps:
            return lr * (step / warmup_steps)
        # 2) If step > steps, return min_lr
        if step > steps:
            return min_lr
        # 3) In between, use cosine decay down to min_lr
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)
    
    # AMP setup
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = device.startswith("cuda")
    
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    print(f"Training on {device} with streaming C4 data")
    print(f"Model: 1024 dim, 4 layers, {len(vocab)} vocab")
    print("=" * 60)
    
    # Separate streams for train and val
    train_stream = char_stream
    _, _, _, val_stream_iter = build_streaming_dataset(seq_len=seq_len)
    
    step_times = []
    start_time = time.time()
    
    for step in range(start_step, steps + 1):
        step_start = time.time()
        
        # Apply warmup LR
        current_lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr
        
        # Sample batch from stream
        x, y = sample_batch(train_stream, batch_size, seq_len, device)
        
        opt.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss at step {step}, skipping")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(opt)
        scaler.update()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 100:
            step_times.pop(0)
        
        # Validation
        if step % val_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vlosses = []
                for _ in range(10):
                    vx, vy = sample_batch(val_stream_iter, batch_size, seq_len, device)
                    with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                    vlosses.append(vloss.item())
                vloss = sum(vlosses) / len(vlosses)
            model.train()
            
            avg_step_time = sum(step_times) / len(step_times)
            eta_seconds = avg_step_time * (steps - step)
            eta_mins = eta_seconds / 60
            elapsed_mins = (time.time() - start_time) / 60
            
            # Calculate total chars processed
            chars_processed = step * batch_size * seq_len
            chars_in_billions = chars_processed / 1e9
            
            print(
                f"[step {step:5d}] train_loss {loss.item():.4f} | val_loss {vloss:.4f} | "
                f"ppl {math.exp(min(vloss, 10.0)):.2f} | {chars_in_billions:.3f}B chars | "
                f"{avg_step_time*1000:.0f}ms/step | ETA {eta_mins:.1f}m"
            )
            
            # Sample generation
            if step % (val_every * 5) == 0:
                model.eval()
                with torch.no_grad():
                    seed = x[:1]
                    gen_tokens = []
                    idx = seed.clone()
                    for _ in range(150):
                        logits = model(idx)
                        last = logits[:, -1, :]
                        probs = F.softmax(last / 0.8, dim=-1)
                        nxt = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat([idx[:, 1:], nxt], dim=1)
                        gen_tokens.append(nxt.item())
                    sample = "".join(itos.get(i, '?') for i in gen_tokens)
                    print(f"Sample: {repr(sample[:120])}...")
                model.train()
        
        # Save checkpoint
        if step % save_every == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'vocab': vocab,
                'stoi': stoi,
                'itos': itos,
            }, checkpoint_path)
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"✓ Checkpoint saved ({timestamp})")
    
    return model, stoi, itos

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    
    print(f"Training on: {dev}")
    
    train_streaming(
        device=dev,
        seq_len=256,
        steps=50000,
        batch_size=64,
        lr=1e-3,
        checkpoint_path="wave_lm_streaming.pt",
        save_every=500,
        val_every=100,
    )
