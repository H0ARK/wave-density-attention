import argparse
import math
import os
import sys
import time
from pathlib import Path
import json

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from wave_dencity.model_thought import WaveCharLM
from wave_dencity.transplant.data import get_data_streamer
from wave_dencity.inference import generate_text


class StepLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.f = open(log_path, "a", encoding="utf-8")

    def log(self, step: int, data: dict):
        entry = {"step": step, "time": time.time(), **data}
        self.f.write(json.dumps(entry) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument(
        "--resume_adapter", type=str, default=None, help="Path to checkpoint to resume"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = json.load(f)

    # Simple namespace for config
    class Config:
        def __init__(self, d):
            self.__dict__.update(d)

    cfg = Config(cfg_dict)

    device = cfg.device
    dtype = (
        torch.bfloat16
        if cfg.torch_dtype == "bfloat16"
        else (torch.float16 if cfg.torch_dtype == "float16" else torch.float32)
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {cfg.teacher_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.teacher_model_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Initializing model (from scratch)...")
    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=cfg.seq_len,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_masks=cfg.wda_num_masks,
        num_waves_per_mask=cfg.wda_num_waves_per_mask,
        topk_masks=cfg.wda_topk_masks,
        attn_alpha=cfg.wda_attn_alpha,
        use_chunks=getattr(cfg, "use_chunks", False),
        chunk_size=getattr(cfg, "chunk_size", 64),
        chunk_reg_weight=getattr(cfg, "chunk_reg_weight", 0.0),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.1f}M")

    start_step = 1
    resume_ckpt = None
    if args.resume_adapter:
        print(f"Loading checkpoint from {args.resume_adapter}...")
        resume_ckpt = torch.load(args.resume_adapter, map_location=device)
        state_dict = resume_ckpt["model_state_dict"]
        # Strip _orig_mod. prefix if it exists in the checkpoint
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                clean_state_dict[k[len("_orig_mod.") :]] = v
            else:
                clean_state_dict[k] = v
        model.load_state_dict(clean_state_dict)
        start_step = resume_ckpt.get("step", 0) + 1
        print(f"Resumed model at step {start_step - 1}")

    # Optimization for 3090: torch.compile
    if getattr(cfg, "use_compile", True):
        if sys.platform == "win32":
            print(
                "Note: torch.compile has limited support on Windows. Skipping to avoid Triton requirement."
            )
        else:
            print("Compiling model (torch.compile)...")
            try:
                model = torch.compile(
                    model, mode=getattr(cfg, "compile_mode", "reduce-overhead")
                )
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

    # Data streamer
    print("Initializing data streamer...")
    stream = get_data_streamer(
        tokenizer,
        cfg.datasets,
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        device=device,
    )

    # Optimization for 3090: Fused AdamW
    opt_kwargs = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    try:
        # Fused is much faster on 3090 if using PT 2.0+
        opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
        print("Using Fused AdamW optimizer.")
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        print("Using standard AdamW optimizer.")

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device == "cuda" and dtype == torch.float16)
    )
    amp_enabled = device == "cuda"

    logger = StepLogger(str(out_dir / "training_log.jsonl"))
    t0 = time.time()
    tokens_seen = 0

    if resume_ckpt and "optimizer_state_dict" in resume_ckpt:
        print("Resuming optimizer state...")
        try:
            opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: Could not resume optimizer state: {e}")

    print(f"Starting training for {cfg.steps} steps from step {start_step}...")
    model.train()

    for step in range(start_step, cfg.steps + 1):
        # LR schedule
        if step <= cfg.warmup_steps:
            curr_lr = cfg.lr * (step / cfg.warmup_steps)
        else:
            curr_lr = cfg.lr

        for g in opt.param_groups:
            g["lr"] = curr_lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_reg = 0.0

        for _ in range(cfg.grad_accum):
            batch = next(stream)
            tokens_seen += int(batch.attention_mask.sum().item())

            with torch.amp.autocast(
                device_type="cuda" if device == "cuda" else "cpu",
                enabled=amp_enabled,
                dtype=dtype,
            ):
                # Forward pass
                if getattr(cfg, "use_chunks", False) and cfg.chunk_reg_weight > 0:
                    logits, reg_loss = model(batch.input_ids, return_chunk_reg=True)
                else:
                    logits = model(batch.input_ids)
                    reg_loss = torch.tensor(0.0, device=device)

                # Loss computation
                # Shift for causal LM
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch.input_ids[:, 1:].contiguous()

                # Apply loss mask if available
                if batch.loss_mask is not None:
                    # loss_mask is [B, S], we need [B, S-1]
                    shift_mask = batch.loss_mask[:, 1:].contiguous()
                    # Flatten and mask
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    flat_mask = shift_mask.view(-1)

                    # We can use CrossEntropy with ignore_index=-100
                    target_labels = flat_labels.clone()
                    target_labels[flat_mask == 0] = -100
                    loss = F.cross_entropy(flat_logits, target_labels)
                else:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                if cfg.chunk_reg_weight > 0:
                    loss = loss + cfg.chunk_reg_weight * reg_loss

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at step {step}")
                continue

            total_loss += loss.item()
            total_reg += reg_loss.item()

            scaler.scale(loss / cfg.grad_accum).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(opt)
        scaler.update()

        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t0
            tok_s = tokens_seen / max(dt, 1e-6)
            mem = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if device == "cuda"
                else 0.0
            )
            avg_loss = total_loss / cfg.grad_accum
            avg_reg = total_reg / cfg.grad_accum
            print(
                f"[{step:5d}/{cfg.steps}] loss={avg_loss:.4f} reg={avg_reg:.4f} lr={curr_lr:.2e} tok/s={tok_s:.0f} vram={mem:.2f}GB"
            )

            logger.log(
                step,
                {
                    "loss": avg_loss,
                    "reg_loss": avg_reg,
                    "lr": curr_lr,
                    "tok_s": tok_s,
                    "vram_gb": mem,
                },
            )

        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt_path = out_dir / f"model_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "cfg": cfg_dict,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        if step % getattr(cfg, "generate_every", 500) == 0 or step == cfg.steps:
            # Generation test
            print("\n" + "=" * 50)
            print(f"GENERATION TEST AT STEP {step}")
            test_prompts = [
                "User: Hello! Who are you?\nAssistant:",
                "The quick brown fox",
                "Python is a programming language that",
            ]
            for prompt in test_prompts:
                try:
                    # Use model.module if it was compiled, otherwise model
                    gen_model = model.module if hasattr(model, "module") else model
                    sample = generate_text(
                        gen_model, tokenizer, prompt, max_tokens=64, device=device
                    )
                    print(f"\nPrompt: {prompt}")
                    print(f"Output: {sample}")
                except Exception as e:
                    print(f"Generation failed for prompt '{prompt}': {e}")
            print("=" * 50 + "\n")
    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
