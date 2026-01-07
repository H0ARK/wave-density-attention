import argparse
import time

import torch
import torch.nn.functional as F

from wave_dencity import WaveCharLM


def load_checkpoint_model(checkpoint_path: str, device: str = "cpu"):
    print(f"Loading {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"]
    model_cfg = ckpt.get("model_cfg", {}) or {}

    vocab_size, embed_dim = state_dict["tok_emb.weight"].shape
    seq_len = state_dict["pos_emb"].shape[1]

    block_indices = {int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks.")}
    num_layers = (max(block_indices) + 1) if block_indices else 1

    num_masks, num_waves, _ = state_dict["blocks.0.freqs.0"].shape

    print(f"Detected Arch: {num_layers} layers, {embed_dim} dim, {seq_len} context, {num_masks} masks, {num_waves} waves")

    tokenizer = None
    stoi = ckpt.get("stoi")
    itos = ckpt.get("itos")
    if ckpt.get("tokenizer") == "gpt2":
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        vocab_size = len(tokenizer)

    model = WaveCharLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_masks=num_masks,
        num_waves_per_mask=num_waves,
        topk_masks=int(model_cfg.get("topk_masks", 8)),
        attn_alpha=float(model_cfg.get("attn_alpha", 3.0)),
        content_rank=int(model_cfg.get("content_rank", 8)),
        content_mix=float(model_cfg.get("content_mix", 0.0)),
        learned_content=bool(model_cfg.get("learned_content", False)),
        use_sin_waves=bool(model_cfg.get("use_sin_waves", True)),
        bias_rank=int(model_cfg.get("bias_rank", 0)),
        bias_mix=float(model_cfg.get("bias_mix", 0.0)),
        cache_kernels=bool(model_cfg.get("cache_kernels", False)),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Note: loaded with strict=False (missing={len(missing)}, unexpected={len(unexpected)})")

    model.to(device).eval()
    return model, tokenizer, stoi, itos


def _best_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def generate(
    model: WaveCharLM,
    tokenizer,
    stoi,
    itos,
    prompt: str,
    steps: int = 200,
    temp: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
):
    seq_len = model.seq_len
    if tokenizer is not None:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        if len(tokens) >= seq_len:
            idx_list = tokens[-seq_len:]
        else:
            idx_list = [pad_id] * (seq_len - len(tokens)) + tokens

        def decode_one(tid: int) -> str:
            return tokenizer.decode([tid])

    else:
        if stoi is None or itos is None:
            raise ValueError("Checkpoint missing tokenizer and stoi/itos; can't decode.")
        prompt_indices = [stoi.get(c, stoi.get(" ", 12)) for c in prompt]
        pad_id = stoi.get(" ", 12)
        idx_list = ([pad_id] * (seq_len - len(prompt_indices)) + prompt_indices)[-seq_len:]

        def decode_one(tid: int) -> str:
            return itos.get(tid, "?")

    idx = torch.tensor([idx_list], dtype=torch.long, device=device)
    print(f"\nPROMPT:\n{prompt}\n" + "-" * 40)

    with torch.inference_mode():
        for _ in range(steps):
            logits = model(idx)[:, -1, :]
            logits = logits / max(temp, 1e-6)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                remove = cumprobs > float(top_p)
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx[:, 1:], nxt], dim=1)
            print(decode_one(int(nxt.item())), end="", flush=True)

    print("\n" + "-" * 40)


@torch.no_grad()
def benchmark_forward(model: WaveCharLM, device: str, iters: int = 50):
    model.eval()
    seq_len = model.seq_len
    vocab = model.vocab_size
    x = torch.randint(0, vocab, (1, seq_len), device=device)
    for _ in range(5):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Forward: {iters/dt:.2f} it/s ({dt/iters*1000:.1f} ms/iter) seq_len={seq_len}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wave_lm_bpe-v5.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cache-kernels", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--prompt", default="User: Hey MOM what's the weather going to be like today?\nAssistant:")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    device = _best_device(args.device)
    print(f"Using device: {device}")

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model, tokenizer, stoi, itos = load_checkpoint_model(args.ckpt, device=device)

    if args.cache_kernels:
        for blk in model.blocks:
            blk.cache_kernels = True
            blk.invalidate_kernel_cache()
        print("Enabled kernel caching.")

    if args.compile and device.startswith("cuda"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    if args.bench:
        benchmark_forward(model, device=device, iters=50)

    generate(
        model,
        tokenizer,
        stoi,
        itos,
        args.prompt,
        steps=args.steps,
        temp=args.temp,
        top_p=args.top_p,
        device=device,
    )


if __name__ == "__main__":
    main()

