from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
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
    # Optional: run for exactly N optimizer steps starting from the current run start.
    # If >0, training stops when `step >= start_step + run_steps` (independent of autopilot progress).
    run_steps: int = 0
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
    handoff_mode: Literal["global", "sequential"] = "global"
    steps_per_layer: int = 100
    handoff_stabilization_steps: int = 0
    handoff_power: float = 1.0
    freeze_converted: bool = False
    converted_lr_scale: float = 0.1
    active_layers: list[int] | None = None  # If set, only these layers are trainable

    # Internal matching weights
    hidden_match_weight: float = 0.0
    attn_match_weight: float = 0.0

    # Feature distillation (modular)
    distill_enabled: bool = False
    distill_targets: list[str] | None = None
    distill_mse_weight: float = 0.0
    distill_cosine_weight: float = 0.0
    distill_layer_weights: Literal["uniform", "depth_decay"] | list[float] | None = None

    # Autopilot (Dynamic Handoff)
    autopilot_enabled: bool = False
    autopilot_kl_low: float = 0.5
    autopilot_kl_high: float = 1.5

    # EMA Teacher (Mean-Teacher stabilization)
    # When enabled, training can switch from an external HF teacher model to a lightweight
    # EMA (exponential moving average) teacher derived from the student parameters.
    # This helps avoid sharp endpoint cliffs (alpha->1, teacher_scale->0).
    ema_teacher_enabled: bool = False
    ema_decay: float = 0.9995
    # Progress at which to start using EMA teacher logits (and optionally offload the external teacher).
    ema_start_progress: float = 0.0
    # Endpoint avoidance caps/floors (never hit exactly 1.0/0.0 during training)
    ema_alpha_cap: float = 0.995
    ema_alpha_cap_end: float | None = None
    ema_alpha_cap_steps: int = 0
    ema_teacher_scale_floor: float = 0.05
    ema_teacher_scale_floor_end: float | None = None
    ema_teacher_scale_floor_steps: int = 0
    # Loss schedule during EMA stage
    ema_kl_weight: float = 0.2
    ema_ce_weight: float = 1.5
    # Routing (gate temperature) during EMA stage
    ema_gate_temp_start: float = 0.08
    ema_gate_temp_end: float = 0.05
    ema_gate_temp_steps: int = 1000

    # Final landing / packaging
    # - teacher_free_mode: do a short CE-only burn-in with the teacher disabled.
    #   This creates a teacher-free artifact suitable for release.
    # - final_force_teacher_off: whether to forcibly set alpha=1, teacher_scale=0 at final save.
    teacher_free_mode: bool = False
    final_force_teacher_off: bool = True

    # Teacher-free burn-in: scale overrides.
    # NOTE: `teacher_attn` inside ParallelAttentionWrapper is the frozen baseline attention module
    # (copied from the original model when patching), not an external teacher model.
    # These overrides control whether a teacher-free run is truly WDA-only (teacher_scale=0)
    # or a hybrid (teacher_scale>0) while still avoiding any external teacher targets.
    teacher_free_alpha: float = 1.0
    teacher_free_teacher_scale: float = 0.0
    # Optional linear ramp for teacher_scale during a teacher-free run.
    # If teacher_free_teacher_scale_end is set and teacher_free_teacher_scale_steps>0,
    # the effective teacher_scale becomes:
    #   ts(step) = ts_start + p * (ts_end - ts_start),  p=clamp((local_step)/steps,0,1)
    # where local_step counts optimizer steps since the run started (after resume).
    teacher_free_teacher_scale_end: float | None = None
    teacher_free_teacher_scale_steps: int = 0
    teacher_free_wda_scale: float = 1.0
    teacher_free_wda_scale_end: float | None = None
    teacher_free_wda_scale_steps: int = 0

    # Teacher-free burn-in helpers
    # In WDA-only mode, negative or near-zero gammas can destabilize behavior.
    # Optionally re-initialize all wrapper gammas at the start of a teacher-free run.
    # teacher_free_gamma_reset:
    # - None: auto (reset only if teacher_free_teacher_scale≈0)
    # - True: always apply init/clamp
    # - False: never touch gammas (preserve checkpoint behavior)
    teacher_free_gamma_reset: bool | None = None
    teacher_free_gamma_init: float | None = 1.0
    teacher_free_gamma_min: float = 0.05

    # Optional per-layer scaling for alpha.
    # - uniform: same alpha on all layers
    # - linear: deeper layers get larger alpha (0..1)
    # - sqrt: gentler version of linear
    alpha_layer_scale: Literal["uniform", "linear", "sqrt"] = "uniform"

    # Gamma initialization and regularization
    init_gamma: float = 0.0
    gamma_l1_weight: float = 0.0
    gamma_l1_target: float = 0.0

    # Gamma/gain-only boost stage
    train_gamma_only: bool = False
    gamma_only_lr: float | None = None

    # Gain transfer: gamma magnitude floor schedule (sign-preserving)
    gamma_floor_start: float = 0.0
    gamma_floor_end: float = 0.0
    gamma_floor_start_step: int = 0
    gamma_floor_steps: int = 0
    gamma_floor_topk: int = 0
    gamma_floor_focus_scale: float = 1.0
    gamma_max: float | None = None
    baseline_pct_log_topk: int = 5
    wda_gain_min: float | None = 0.0
    wda_gain_max: float | None = None
    wda_gain_l2_weight: float = 0.0
    wda_gain_target: float = 1.0

    # WDA params
    wda_num_masks: int = 32
    wda_num_waves_per_mask: int = 12
    wda_topk_masks: int = 8
    wda_gate_temp: float = 1.0
    wda_attn_alpha: float = 3.0
    wda_content_mix: float = 0.15
    wda_learned_content: bool = True
    wda_use_sin_waves: bool = True
    wda_use_sampling: bool = False
    wda_num_samples: int = 64
    wda_noise_sigma: float = 0.12
    wda_step_alpha: float = 6.0
    wda_use_checkpoint: bool = False
    routing_entropy_weight: float = 0.0

    # Schedule/plateau stages
    schedule_stages: list[dict[str, Any]] | None = None

    # Pipeline feature flags (optional)
    # - run_mode: mixing_bridge | teacher_target_distill | teacher_free
    # - teacher_type: external_hf | ema_student | none
    # - student_attn_impl: parallel_mixer | wda | baseline
    run_mode: (
        Literal["mixing_bridge", "teacher_target_distill", "teacher_free"] | None
    ) = None
    teacher_type: Literal["external_hf", "ema_student", "none"] | None = None
    student_attn_impl: Literal["parallel_mixer", "wda", "baseline"] | None = None

    # Runtime
    device: Literal["cuda", "mps", "cpu"] | str = "cuda"
    torch_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    quantization: Literal["4bit", "8bit"] | None = None
    gradient_checkpointing: bool = True
    loss_chunk_size: int = 32

    # Optimizer
    use_fused_adamw: bool = False

    # Logging/checkpointing
    log_every: int = 10
    save_every: int = 200
    out_dir: str = "private/checkpoints/transplant_gemma270m"
    log_jsonl_path: str | None = None
    log_fields_include: list[str] | None = None
    log_fields_exclude: list[str] | None = None
    log_sqlite_path: str | None = None
    log_sqlite_table: str = "training_log"
    log_sqlite_every: int = 1
    log_drop_defaults: bool = True
    log_drop_values: list[float] | None = None
    log_drop_tolerance: float = 1e-6

    # Eval harness
    eval_enabled: bool = False
    eval_every: int = 0
    eval_batches: int = 1
    eval_prompts: list[str] | None = None
    eval_max_new_tokens: int = 128
    eval_temperature: float = 0.7
    eval_top_p: float = 0.9
    eval_teacher_free: bool = False
    eval_out_dir: str | None = None


def _normalize_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)

    run_cfg = data.pop("run", None) or {}
    teacher_cfg = data.pop("teacher", None) or {}
    student_cfg = data.pop("student", None) or {}
    logging_cfg = data.pop("logging", None) or {}
    distill_cfg = data.pop("distill", None) or {}
    schedule_cfg = data.pop("schedule", None) or {}
    eval_cfg = data.pop("eval", None) or {}

    if "mode" in run_cfg and "run_mode" not in data:
        data["run_mode"] = run_cfg["mode"]
    if "type" in teacher_cfg and "teacher_type" not in data:
        data["teacher_type"] = teacher_cfg["type"]
    if "attn_impl" in student_cfg and "student_attn_impl" not in data:
        data["student_attn_impl"] = student_cfg["attn_impl"]

    if "include" in logging_cfg and "log_fields_include" not in data:
        data["log_fields_include"] = logging_cfg["include"]
    if "exclude" in logging_cfg and "log_fields_exclude" not in data:
        data["log_fields_exclude"] = logging_cfg["exclude"]
    if "fields" in logging_cfg and "log_fields_include" not in data:
        data["log_fields_include"] = logging_cfg["fields"]
    if "jsonl_path" in logging_cfg and "log_jsonl_path" not in data:
        data["log_jsonl_path"] = logging_cfg["jsonl_path"]
    if "sqlite_path" in logging_cfg and "log_sqlite_path" not in data:
        data["log_sqlite_path"] = logging_cfg["sqlite_path"]
    if "sqlite_table" in logging_cfg and "log_sqlite_table" not in data:
        data["log_sqlite_table"] = logging_cfg["sqlite_table"]
    if "sqlite_every" in logging_cfg and "log_sqlite_every" not in data:
        data["log_sqlite_every"] = logging_cfg["sqlite_every"]
    if "drop_defaults" in logging_cfg and "log_drop_defaults" not in data:
        data["log_drop_defaults"] = logging_cfg["drop_defaults"]
    if "drop_values" in logging_cfg and "log_drop_values" not in data:
        data["log_drop_values"] = logging_cfg["drop_values"]
    if "drop_tolerance" in logging_cfg and "log_drop_tolerance" not in data:
        data["log_drop_tolerance"] = logging_cfg["drop_tolerance"]

    if "enabled" in distill_cfg and "distill_enabled" not in data:
        data["distill_enabled"] = distill_cfg["enabled"]
    if "targets" in distill_cfg and "distill_targets" not in data:
        data["distill_targets"] = distill_cfg["targets"]
    if "mse_weight" in distill_cfg and "distill_mse_weight" not in data:
        data["distill_mse_weight"] = distill_cfg["mse_weight"]
    if "cosine_weight" in distill_cfg and "distill_cosine_weight" not in data:
        data["distill_cosine_weight"] = distill_cfg["cosine_weight"]
    if "layer_weights" in distill_cfg and "distill_layer_weights" not in data:
        data["distill_layer_weights"] = distill_cfg["layer_weights"]

    if "stages" in schedule_cfg and "schedule_stages" not in data:
        data["schedule_stages"] = schedule_cfg["stages"]

    if "enabled" in eval_cfg and "eval_enabled" not in data:
        data["eval_enabled"] = eval_cfg["enabled"]
    if "every" in eval_cfg and "eval_every" not in data:
        data["eval_every"] = eval_cfg["every"]
    if "batches" in eval_cfg and "eval_batches" not in data:
        data["eval_batches"] = eval_cfg["batches"]
    if "prompts" in eval_cfg and "eval_prompts" not in data:
        data["eval_prompts"] = eval_cfg["prompts"]
    if "max_new_tokens" in eval_cfg and "eval_max_new_tokens" not in data:
        data["eval_max_new_tokens"] = eval_cfg["max_new_tokens"]
    if "temperature" in eval_cfg and "eval_temperature" not in data:
        data["eval_temperature"] = eval_cfg["temperature"]
    if "top_p" in eval_cfg and "eval_top_p" not in data:
        data["eval_top_p"] = eval_cfg["top_p"]
    if "teacher_free" in eval_cfg and "eval_teacher_free" not in data:
        data["eval_teacher_free"] = eval_cfg["teacher_free"]
    if "out_dir" in eval_cfg and "eval_out_dir" not in data:
        data["eval_out_dir"] = eval_cfg["out_dir"]

    allowed = {f.name for f in fields(TransplantConfig)}
    unknown = sorted(k for k in data.keys() if k not in allowed)
    if unknown:
        print(f"WARNING: Unknown config keys ignored: {', '.join(unknown)}")

    return {k: v for k, v in data.items() if k in allowed}


def load_config(path: str | Path) -> TransplantConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = _normalize_config_dict(data)
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
