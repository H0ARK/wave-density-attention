from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class TransplantConfig:
    # Teacher
    teacher_model_id: str = "dapn/gemma-3-270M-it-coder"

    # Data
    dataset_path: str = "private/data/c4_clean.txt"
    datasets: list[dict[str, Any]] | None = None
    seq_len: int = 1024
    micro_batch_size: int = 1
    grad_accum: int = 16

    # Training
    steps: int = 2000
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0

    # Distillation
    temperature: float = 2.0
    kl_weight: float = 1.0
    ce_weight: float = 0.0

    # Alpha schedule (teacher + alpha * gamma * WDA)
    alpha_max: float = 1.0
    alpha_warmup_steps: int = 500

    # Handoff schedule (disable teacher over time)
    # teacher_scale is applied to the teacher attention output.
    # wda_scale is applied to (alpha * gamma * wda_out).
    # Defaults keep the teacher fully on and the WDA branch enabled.
    teacher_scale_start: float = 1.0
    teacher_scale_end: float = 1.0
    wda_scale_start: float = 1.0
    wda_scale_end: float = 1.0
    handoff_start_step: int = 0
    handoff_steps: int = 0

    # Optional per-layer scaling for alpha.
    # - uniform: same alpha on all layers
    # - linear: deeper layers get larger alpha (0..1)
    # - sqrt: gentler version of linear
    alpha_layer_scale: Literal["uniform", "linear", "sqrt"] = "uniform"

    # Gamma initialization and regularization
    init_gamma: float = 0.0
    gamma_l1_weight: float = 0.0
    gamma_l1_target: float = 0.0

    # WDA params
    wda_num_masks: int = 32
    wda_num_waves_per_mask: int = 12
    wda_topk_masks: int = 8
    wda_attn_alpha: float = 3.0
    wda_content_mix: float = 0.15
    wda_learned_content: bool = True
    wda_use_sin_waves: bool = True
    wda_use_sampling: bool = False
    wda_num_samples: int = 64
    wda_noise_sigma: float = 0.12
    wda_step_alpha: float = 6.0
    wda_use_checkpoint: bool = False

    # Runtime
    device: Literal["cuda", "mps", "cpu"] | str = "cuda"
    torch_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    quantization: Literal["4bit", "8bit"] | None = None
    gradient_checkpointing: bool = True

    # Logging/checkpointing
    log_every: int = 10
    save_every: int = 200
    out_dir: str = "private/checkpoints/transplant_gemma270m"


def load_config(path: str | Path) -> TransplantConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return TransplantConfig(**data)


def save_config(cfg: TransplantConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)


def config_to_pretty_json(cfg: TransplantConfig) -> str:
    return json.dumps(asdict(cfg), indent=2, sort_keys=True)


def parse_torch_dtype(dtype_str: str):
    import torch

    s = str(dtype_str).lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {dtype_str}")
