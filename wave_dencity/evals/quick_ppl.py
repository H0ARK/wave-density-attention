import argparse
import math

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from wave_dencity import (
    WaveCharLM,
    sample_batch,
    build_streaming_dataset,
    build_streaming_ultrachat_dataset,
)


def _best_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _strip_compile_prefix(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def eval_ppl(
    model: WaveCharLM,
    tokenizer: GPT2TokenizerFast,
    dataset: str,
    data_cfg: dict,
    device: str,
    batches: int,
    batch_size: int,
    split: str | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    seq_len = model.seq_len

    if dataset == "c4":
        stream = build_streaming_dataset(tokenizer, seq_len=seq_len)
        assistant_only = False
    elif dataset == "ultrachat":
        split_name = split or str(data_cfg.get("val_split", "test_sft"))
        stream = build_streaming_ultrachat_dataset(
            tokenizer,
            split=split_name,
            dataset_name=str(data_cfg.get("dataset_name", "HuggingFaceH4/ultrachat_200k")),
            assistant_only_loss=True,
            include_assistant_prefix_in_loss=bool(data_cfg.get("include_assistant_prefix_in_loss", False)),
        )
        assistant_only = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model.eval()
    losses: list[float] = []
    for i in range(batches):
        x, y = sample_batch(
            stream,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            assistant_only_loss=assistant_only,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        losses.append(float(loss.item()))
        if verbose and (i + 1) % max(1, batches // 4) == 0:
            avg = sum(losses) / len(losses)
            ppl = math.exp(min(avg, 10.0))
            print(f"  {dataset}: {i+1:>3}/{batches} avg_loss={avg:.4f} ppl={ppl:.2f}")

    avg = sum(losses) / len(losses)
    ppl = math.exp(min(avg, 10.0))
    return avg, ppl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to .pt checkpoint")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dataset",
        default="trained",
        choices=["trained", "c4", "ultrachat", "both"],
        help="Which dataset to evaluate on (default: use checkpoint's data_cfg.dataset)",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="Dataset split for UltraChat eval (default: checkpoint data_cfg.val_split)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat evaluation multiple times to show streaming/shuffle variance.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-batch progress output (useful with --repeats).",
    )
    args = ap.parse_args()

    device = args.device or _best_device()
    print("Device:", device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = _strip_compile_prefix(ckpt["model"])
    model_cfg = ckpt.get("model_cfg", {}) or {}
    data_cfg = ckpt.get("data_cfg", {}) or {}

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Streaming datasets may include long examples; we only sample fixed windows,
    # so avoid noisy warnings about >1024 token sequences.
    tokenizer.model_max_length = 10**9

    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=256,
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

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict strict=False -> missing={len(missing)} unexpected={len(unexpected)}")
    print("tie_embeddings:", model.tie_embeddings)
    print("tok_emb == head?", model.tok_emb.weight.data_ptr() == model.head.weight.data_ptr())

    trained_on = str(data_cfg.get("dataset", "c4")).lower()
    if trained_on not in ("c4", "ultrachat"):
        trained_on = "c4"

    print("Checkpoint dataset:", data_cfg.get("dataset"))

    if args.dataset == "trained":
        datasets = [trained_on]
    elif args.dataset == "both":
        datasets = ["ultrachat", "c4"]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        print("Evaluating:", ds)
        results: list[tuple[float, float]] = []
        for r in range(args.repeats):
            if args.repeats > 1:
                print(f"  repeat {r+1}/{args.repeats}")
            loss, ppl = eval_ppl(
                model,
                tokenizer,
                ds,
                data_cfg,
                device,
                batches=args.batches,
                batch_size=args.batch_size,
                split=args.split,
                verbose=not args.quiet,
            )
            results.append((loss, ppl))

        if len(results) == 1:
            loss, ppl = results[0]
            print(f"FINAL {ds}: loss={loss:.4f} ppl={ppl:.2f}")
        else:
            losses = [x for x, _ in results]
            ppls = [p for _, p in results]
            print(
                f"FINAL {ds}: ppl min={min(ppls):.2f} mean={sum(ppls)/len(ppls):.2f} max={max(ppls):.2f} "
                f"(loss mean={sum(losses)/len(losses):.4f})"
            )


if __name__ == "__main__":
    main()
