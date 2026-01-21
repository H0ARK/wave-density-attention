import math
import time
import os
import sys
import datetime

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from wave_dencity import (
    WaveCharLM,
    build_streaming_dataset,
    build_streaming_ultrachat_dataset,
    sample_batch,
    sample_mixed_batch,
    generate_text,
)


def train_streaming_lm(
    device: str = "cpu",
    seq_len: int = 256,
    steps: int = 10000,
    batch_size: int = 64,
    lr: float = 1e-3,
    see: int = 0,
    val_every: int = 100,
    val_batches: int = 10,
    checkpoint_path: str = "wave_lm_bpe.pt",
    save_every: int = 500,
    use_8bit_opt: bool = False,
    model_cfg: dict | None = None,
    data_cfg: dict | None = None,
    train_cfg: dict | None = None,
):
    """Train wave-density LM with streaming C4 and BPE tokenization."""
    torch.manual_seed(see)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Vocab size: {len(tokenizer)}")

    model_cfg = model_cfg or {}
    data_cfg = data_cfg or {}
    train_cfg = train_cfg or {}

    # Data config
    buffer_size = int(data_cfg.get("buffer_size", 10000))
    dataset_kind = str(data_cfg.get("dataset", "c4")).lower()
    dataset_split = str(
        data_cfg.get("split", "train_sft" if dataset_kind == "ultrachat" else "train")
    )
    dataset_val_split = str(
        data_cfg.get(
            "val_split", "test_sft" if dataset_kind == "ultrachat" else "train"
        )
    )
    assistant_only_loss = bool(
        data_cfg.get(
            "assistant_only_loss",
            dataset_kind == "shared_train" if dataset_kind == "ultrachat" else False,
        )
    )  # Simplified
    if dataset_kind == "ultrachat":
        assistant_only_loss = data_cfg.get("assistant_only_loss", True)

    include_assistant_prefix_in_loss = bool(
        data_cfg.get("include_assistant_prefix_in_loss", True)
    )
    dataset_name = str(
        data_cfg.get(
            "dataset_name",
            (
                "HuggingFaceH4/ultrachat_200k"
                if dataset_kind == "ultrachat"
                else "allenai/c4"
            ),
        )
    )

    # Training config
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    clip_grad = float(train_cfg.get("clip_grad", 0.5))
    use_compile = bool(train_cfg.get("use_compile", True))
    compile_mode = str(train_cfg.get("compile_mode", "reduce-overhead"))
    use_bf16 = bool(train_cfg.get("use_bf16", True))

    use_mixed_phase = bool(train_cfg.get("use_mixed_phase", False))
    mixed_phase_ratio = float(train_cfg.get("mixed_phase_ratio", 0.1))
    mixed_phase_primary_ratio = float(train_cfg.get("mixed_phase_primary_ratio", 0.8))
    mixed_phase_start_step = int(steps * (1.0 - mixed_phase_ratio))

    if dataset_kind == "ultrachat":
        print(
            f"Initializing streaming UltraChat dataset ({dataset_name}, split={dataset_split})..."
        )
        train_stream = build_streaming_ultrachat_dataset(
            tokenizer,
            split=dataset_split,
            buffer_size=buffer_size,
            dataset_name=dataset_name,
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
        )
        val_stream = build_streaming_ultrachat_dataset(
            tokenizer,
            split=dataset_val_split,
            buffer_size=buffer_size,
            dataset_name=dataset_name,
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
        )
        if use_mixed_phase:
            print("Initializing C4 stream for mixed training phase...")
            train_stream_c4 = build_streaming_dataset(
                tokenizer, seq_len=seq_len, buffer_size=buffer_size
            )
        else:
            train_stream_c4 = None
    else:
        print("Initializing streaming C4 dataset...")
        train_stream = build_streaming_dataset(
            tokenizer, seq_len=seq_len, buffer_size=buffer_size
        )
        val_stream = build_streaming_dataset(
            tokenizer, seq_len=seq_len, buffer_size=buffer_size
        )
        train_stream_c4 = None

    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=seq_len,
        embed_dim=int(model_cfg.get("embed_dim", 768)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_masks=int(model_cfg.get("num_masks", 16)),
        num_waves_per_mask=int(model_cfg.get("num_waves_per_mask", 8)),
        topk_masks=int(model_cfg.get("topk_masks", 8)),
        attn_alpha=float(model_cfg.get("attn_alpha", 3.0)),
        content_rank=int(model_cfg.get("content_rank", 8)),
        content_mix=float(model_cfg.get("content_mix", 0.0)),
        learned_content=bool(model_cfg.get("learned_content", False)),
        use_sin_waves=bool(model_cfg.get("use_sin_waves", True)),
        ffn_mult=int(model_cfg.get("ffn_mult", 4)),
        tie_embeddings=bool(model_cfg.get("tie_embeddings", False)),
    ).to(device)

    if device.startswith("cuda") and use_compile:
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"✅ torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"⚠️ torch.compile unavailable, continuing without it: {e}")

    if use_8bit_opt:
        if not device.startswith("cuda"):
            print(
                "Warning: bitsandbytes 8-bit optimizer requires CUDA — falling back to AdamW."
            )
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            try:
                from bitsandbytes.optim import Adam8bit

                print("Using bitsandbytes Adam8bit optimizer")
                opt = Adam8bit(model.parameters(), lr=lr, weight_decay=1e-4)
            except Exception as e:
                print(
                    "Warning: bitsandbytes not available or failed to initialize (falling back to AdamW)."
                )
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step / max(warmup_steps, 1))
        return lr

    use_amp = device.startswith("cuda")
    if device.startswith("cuda"):
        amp_device = "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    elif device == "mps":
        amp_device = "cpu"
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        use_amp = False
    else:
        amp_device = "cpu"
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    start_step = 1
    best_val_loss = float("inf")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        try:
            opt.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            print("✅ Optimizer state loaded successfully")
        except (ValueError, KeyError) as e:
            print(
                f"⚠️ Could not load optimizer state (likely due to model changes): {e}"
            )

        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step-1}, best_val_loss={best_val_loss:.4f}")

    print("\n🚀 TRAINING START")
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    step_times = []
    start_time = time.time()

    for step in range(start_step, steps + 1):
        step_start = time.time()
        current_lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr

        in_mixed_phase = (
            use_mixed_phase
            and step >= mixed_phase_start_step
            and train_stream_c4 is not None
        )

        if in_mixed_phase:
            x, y = sample_mixed_batch(
                train_stream,
                train_stream_c4,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                mix_ratio=mixed_phase_primary_ratio,
                assistant_only_loss1=assistant_only_loss,
                assistant_only_loss2=False,
            )
        else:
            x, y = sample_batch(
                train_stream,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                assistant_only_loss=assistant_only_loss,
            )

        opt.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast(
            device_type=amp_device, enabled=use_amp, dtype=amp_dtype
        ):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss at step {step}")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt)
        scaler.update()

        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 100:
            step_times.pop(0)

        if step % val_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vlosses = []
                for _ in range(val_batches):
                    vx, vy = sample_batch(
                        val_stream,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=device,
                        assistant_only_loss=assistant_only_loss,
                    )
                    with torch.amp.autocast(
                        device_type=amp_device, enabled=use_amp, dtype=amp_dtype
                    ):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(
                            vlogits.view(-1, vlogits.size(-1)),
                            vy.view(-1),
                            ignore_index=-100,
                        )
                    vlosses.append(vloss.item())
                vloss = float(sum(vlosses) / len(vlosses))

            model.train()
            if vloss < best_val_loss:
                best_val_loss = vloss
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "tokenizer": "gpt2",
                    },
                    checkpoint_path.replace(".pt", "_best.pt"),
                )

            avg_step_time = sum(step_times) / len(step_times)
            eta_mins = (avg_step_time * (steps - step)) / 60
            tokens_processed = step * batch_size * seq_len

            print(
                f"[{step:5d}] train {loss.item():.4f} | val {vloss:.4f} | ppl {math.exp(min(vloss, 10.0)):.2f} | {tokens_processed/1e9:.3f}B tok | {avg_step_time*1000:.0f}ms/step | ETA {eta_mins:.1f}m"
            )

            if step % (val_every * 5) == 0:
                prompt_mom = "User: Hey MOM what's the weather going to be like today?\nAssistant:"
                sample = generate_text(model, tokenizer, prompt_mom, device=device)
                print(f"   Sample: {repr(sample[:100])}...")

        if step % save_every == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "tokenizer": "gpt2",
                },
                checkpoint_path,
            )

    print("\n🎉 TRAINING COMPLETE")
    return model, tokenizer


if __name__ == "__main__":
    dev = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Training on device: {dev}")

    RUN = {
        "seq_len": 256,
        "steps": 10000,
        "batch_size": 32,
        "lr": 2e-4,
        "see": 42,
        "val_every": 100,
        "val_batches": 10,
        "save_every": 500,
        "checkpoint_path": "wda_130m-mom.pt",
        "use_8bit_opt": False,
    }

    MODEL_CFG = {
        "embed_dim": 768,
        "num_layers": 8,
        "num_heads": 4,
        "num_masks": 16,
        "num_waves_per_mask": 8,
        "topk_masks": 8,
        "attn_alpha": 3.0,
        "content_rank": 8,
        "content_mix": 0.15,
        "learned_content": True,
        "use_sin_waves": False,
        "ffn_mult": 4,
        "tie_embeddings": False,
    }

    DATA_CFG = {
        "dataset": "ultrachat",
        "dataset_name": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "val_split": "test_sft",
        "buffer_size": 10000,
        "assistant_only_loss": True,
        "include_assistant_prefix_in_loss": False,
    }

    TRAIN_CFG = {
        "warmup_steps": 500,
        "clip_grad": 0.5,
        "use_compile": True,
        "compile_mode": "reduce-overhead",
        "use_bf16": True,
        "use_mixed_phase": True,
        "mixed_phase_ratio": 0.1,
        "mixed_phase_primary_ratio": 0.8,
    }

    train_streaming_lm(
        device=dev, **RUN, model_cfg=MODEL_CFG, data_cfg=DATA_CFG, train_cfg=TRAIN_CFG
    )
