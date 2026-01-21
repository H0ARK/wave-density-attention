import math
import time
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from wave_dencity import WaveCharLM


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    print("device", device)

    seed = int(os.environ.get("WDA_SWEEP_SEED", "0"))
    torch.manual_seed(seed)
    print("seed", seed)

    path = "private/data/shakespeare.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print("chars", len(text))

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 1_000_000

    ids = tok.encode(text, add_special_tokens=False)
    print("tokens", len(ids))

    split = int(0.9 * len(ids))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)

    seq_len = int(os.environ.get("WDA_SWEEP_SEQ", "128"))
    batch_size = int(os.environ.get("WDA_SWEEP_BATCH", "8"))
    train_steps = int(os.environ.get("WDA_SWEEP_STEPS", "120"))
    val_batches = int(os.environ.get("WDA_SWEEP_VAL", "20"))

    def sample_batch_1d(arr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(arr.numel())
        starts = torch.randint(0, n - (seq_len + 1), (batch_size,))
        x = torch.stack([arr[s : s + seq_len] for s in starts]).to(device)
        y = torch.stack([arr[s + 1 : s + seq_len + 1] for s in starts]).to(device)
        return x, y

    def run_config(num_masks: int, num_waves: int) -> tuple[float, float]:
        model = WaveCharLM(
            vocab_size=len(tok),
            seq_len=seq_len,
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            num_masks=num_masks,
            num_waves_per_mask=num_waves,
            topk_masks=min(4, num_masks),
            attn_alpha=3.0,
            content_rank=8,
            content_mix=0.15,
            learned_content=True,
            use_sin_waves=False,
            ffn_mult=4,
            tie_embeddings=False,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

        model.train()
        t0 = time.time()
        for step in range(1, train_steps + 1):
            x, y = sample_batch_1d(train_ids)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt.step()
            if step in (1, train_steps // 6, train_steps // 2, train_steps):
                dt = time.time() - t0
                print(f"  step {step:4d} loss {loss.item():.4f} ({dt:.1f}s)")

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(val_batches):
                x, y = sample_batch_1d(val_ids)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                losses.append(float(loss.item()))

        v = float(sum(losses) / len(losses))
        ppl = math.exp(min(v, 20.0))
        return v, ppl

    # A small sweep (short runs are noisy; look for big inflections).
    # You can override steps/batch/seq via env vars.
    configs = [
        (16, 4),
        (8, 4),
        (8, 2),
        (4, 2),
        (2, 2),
        (2, 1),
        (1, 1),
    ]

    results: list[tuple[int, int, float, float]] = []
    for m, w in configs:
        print(f"\n=== masks={m} waves={w} ===")
        vloss, ppl = run_config(m, w)
        print("val_loss", f"{vloss:.4f}", "ppl", f"{ppl:.1f}")
        results.append((m, w, vloss, ppl))

    print("\nSUMMARY")
    for m, w, v, ppl in results:
        print(f"masks={m:2d} waves={w:2d}  val_loss={v:.4f}  ppl={ppl:.1f}")


if __name__ == "__main__":
    main()
