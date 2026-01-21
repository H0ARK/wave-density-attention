from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from wave_dencity.transplant.adapter import set_parallel_attn_mode
from wave_dencity.transplant.data import get_data_streamer
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper


def _snapshot_scales(
    wrappers: list[ParallelAttentionWrapper],
) -> list[tuple[float, float, float]]:
    snap = []
    for m in wrappers:
        a = float(m.alpha.item())
        ts = float(m.teacher_scale.item())
        ws = float(m.wda_scale.item())
        snap.append((a, ts, ws))
    return snap


def _restore_scales(
    wrappers: list[ParallelAttentionWrapper], snap: list[tuple[float, float, float]]
) -> None:
    for m, (a, ts, ws) in zip(wrappers, snap):
        m.set_alpha(a)
        m.set_scales(teacher_scale=ts, wda_scale=ws)


def run_eval(
    *,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cfg: Any,
    device: str,
    step: int,
    out_dir: Path,
) -> dict[str, Any]:
    eval_dir = (
        Path(cfg.eval_out_dir)
        if getattr(cfg, "eval_out_dir", None)
        else out_dir / "eval"
    )
    if not eval_dir.is_absolute():
        eval_dir = out_dir / eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    wrappers = [m for m in model.modules() if isinstance(m, ParallelAttentionWrapper)]
    snap = _snapshot_scales(wrappers)

    if getattr(cfg, "eval_teacher_free", False):
        set_parallel_attn_mode(
            model,
            alpha=1.0,
            teacher_scale=0.0,
            wda_scale=1.0,
            gate_temp=getattr(cfg, "wda_gate_temp", None),
        )
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False

    model.eval()

    prompts = list(
        getattr(cfg, "eval_prompts", None)
        or [
            "Explain why the sky is blue in two sentences.",
            "Q: Who wrote Pride and Prejudice? A:",
            "Write a short paragraph describing how to bake sourdough bread.",
        ]
    )

    generations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(getattr(cfg, "eval_max_new_tokens", 128)),
                do_sample=True,
                temperature=float(getattr(cfg, "eval_temperature", 0.7)),
                top_p=float(getattr(cfg, "eval_top_p", 0.9)),
                use_cache=False,
            )
        generations.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(out[0], skip_special_tokens=True),
            }
        )

    eval_batches = int(getattr(cfg, "eval_batches", 1) or 1)
    dataset_cfg = (
        cfg.datasets if getattr(cfg, "datasets", None) is not None else cfg.dataset_path
    )
    eval_stream = get_data_streamer(
        tokenizer,
        dataset_cfg=dataset_cfg,
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        device=device,
    )

    ce_sum = 0.0
    tok_count = 0
    t0 = time.time()
    with torch.no_grad():
        for _ in range(eval_batches):
            batch = next(eval_stream)
            out = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            logits = out.logits[:, :-1, :].float()
            targets = batch.input_ids[:, 1:]
            mask = batch.attention_mask[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            loss = loss.reshape(targets.shape) * mask
            ce_sum += loss.sum().item()
            tok_count += mask.sum().item()

    ce_avg = ce_sum / max(1.0, tok_count)
    dt = time.time() - t0

    metrics = {
        "step": step,
        "eval_ce": ce_avg,
        "eval_tokens": int(tok_count),
        "eval_time_s": dt,
        "prompts": generations,
    }

    out_path = eval_dir / f"eval_step{step}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _restore_scales(wrappers, snap)
    model.train()

    return metrics
