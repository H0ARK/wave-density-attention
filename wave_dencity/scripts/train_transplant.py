import argparse
import math
import os
import sys
import time
from pathlib import Path
import re

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import warnings
import inspect

try:
    # PyTorch 2.x
    from torch.nn.utils.stateless import functional_call as _functional_call
except Exception:  # pragma: no cover
    _functional_call = None

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*None of the inputs have requires_grad=True.*",
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from wave_dencity.transplant.adapter import (
    save_adapter,
    save_adapter_state,
    load_adapter,
)
from wave_dencity.transplant.config import (
    load_config,
    parse_torch_dtype,
    TransplantConfig,
)
from wave_dencity.transplant.data import stream_text_blocks
from wave_dencity.transplant.feature_flags import apply_feature_flags
from wave_dencity.transplant.feature_extractor import FeatureExtractor
from wave_dencity.transplant.instrumentation import (
    collect_baseline_stats,
    summarize_baseline,
)
from wave_dencity.transplant.loss_builder import compute_chunked_logits_losses
from wave_dencity.transplant.loss_builder_v2 import compute_feature_distill_loss
from wave_dencity.transplant.loggers import StepLogger
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.schedule_manager import ScheduleManager
from wave_dencity.transplant.teacher_provider import TeacherProvider
from wave_dencity.transplant.wda import WDABridge
from wave_dencity.transplant.eval_harness import run_eval

import json


class Autopilot:
    """Adaptive scheduler that adjusts handoff progress based on training health.

    If KL divergence is low, the model is adapting well, so we speed up.
    If KL is high, we slow down to avoid divergence.
    """

    def __init__(
        self,
        target_low: float,
        target_high: float,
        ema_alpha: float = 0.05,
        start_progress: float = 0.0,
    ):
        self.target_low = target_low
        self.target_high = target_high
        self.ema_alpha = ema_alpha
        self.current_ema = None
        self.speed = 1.0
        self.progress = float(start_progress)

    def update(self, current_kl: float) -> tuple[float, float]:
        if self.current_ema is None:
            self.current_ema = current_kl
        else:
            self.current_ema = (
                self.ema_alpha * current_kl + (1 - self.ema_alpha) * self.current_ema
            )

        # Proportional adjustment:
        if self.current_ema > self.target_high:
            # Severity: how many times over the limit are we?
            # If we are vastly over (severity > 2), we slam the brakes.
            severity = min(10.0, self.current_ema / self.target_high)
            reduction = (
                0.90**severity
            )  # More aggressive reduction (10% per severity unit)
            self.speed = max(0.001, self.speed * reduction)
        elif self.current_ema < self.target_low:
            # Re-acceleration: faster when speed is very low, then taper.
            # This helps avoid the 'death crawl' at 0.001.
            recovery_factor = 1.05 if self.speed < 0.1 else 1.02
            self.speed = min(2.0, self.speed * recovery_factor)

        # Progress moves by speed
        self.progress += self.speed
        return self.progress, self.speed


def reset_student_gammas(model: torch.nn.Module, autopilot=None):
    """Safety reset: set all gamma parameters to a small stable value to recover, and back off progress."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ParallelAttentionWrapper):
                # Reset to a small stable value instead of 0.0 to prevent total signal loss at alpha=1
                m.gamma.fill_(0.05)

    msg = ">>> SAFETY: Divergence detected. Soft-resetting gammas to 0.05 to recover."
    if autopilot:
        # Subtract progress to force alpha back down if it was high
        # UPGRADED: If we are in Phase C (post-3500), drop back to Safe Harbor (3400)
        if autopilot.progress > 3500:
            autopilot.progress = 3400.0
            msg += " (Phase C Emergency: Forced back to Safe Harbor 3400)"
        else:
            penalty = 200.0
            autopilot.progress = max(0.0, autopilot.progress - penalty)
            msg += f" (Progress penalized by {penalty:.0f})"

        autopilot.speed = 0.005  # Force slow crawl

    print(msg)


# Ensure stdout is unbuffered for robust logging in PowerShell/Redirection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _load_causal_lm(
    model_id: str, *, dtype: torch.dtype, device: str, quantization: str | None = None
):
    kwargs = {"torch_dtype": dtype}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif quantization == "8bit":
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"

    # Gemma3 recommends eager attention for training.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", trust_remote_code=True, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, **kwargs
        )

    if quantization is None:
        model = model.to(device)
    return model


def _get_num_heads(cfg_obj) -> int:
    for key in ["num_attention_heads", "num_heads", "n_head"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_heads from model.config")


def _get_num_layers(cfg_obj) -> int:
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_layers from model.config")


def _alpha_for_step(step: int, *, alpha_max: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(alpha_max)
    return float(alpha_max) * min(1.0, step / float(warmup_steps))


def _ramp(step: int, *, start_step: int, steps: int) -> float:
    if steps <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (step - start_step) / float(steps))))


def _layer_alpha_scale(layer_idx: int, num_layers: int, mode: str) -> float:
    if num_layers <= 1:
        return 1.0
    t = layer_idx / float(num_layers - 1)
    if mode == "uniform":
        return 1.0
    if mode == "linear":
        return float(t)
    if mode == "sqrt":
        return float(math.sqrt(max(t, 0.0)))
    raise ValueError(f"Unknown alpha_layer_scale: {mode}")


_LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]


def _infer_layer_index(path: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        m = pat.search(path)
        if m:
            return int(m.group(1))
    return None


def _set_scales(
    model: torch.nn.Module,
    *,
    step: int,
    cfg: TransplantConfig,
    num_layers: int,
    alpha: float,
    wrappers: list[ParallelAttentionWrapper] | None = None,
) -> dict[int, float]:
    """Sets teacher and wda scales for all layers. Returns dict of {layer_idx: teacher_scale}."""
    layer_teacher_scales = {}

    # Stability Fix: Smoothly ramp down the teacher anchor after the Safe Harbor (prog=3500)
    # instead of dropping it abruptly.
    if step <= 3500:
        min_ts = 0.15 if alpha < 0.99 else 0.0
    else:
        # Phase C: Final Anchor Release (3500 to 5500) - Stretched to 2k steps
        # UPGRADE: Reach absolute 0.0/1.0 at the finish line, but use
        # a microscopic anchor (1e-12) during the transition to keep the math continuous.
        p_release = float(max(0.0, min(1.0, (step - 3500) / 2000.0)))
        min_ts = 0.15 * (1.0 - p_release)

        if p_release < 1.0:
            min_ts = max(min_ts, 1e-12)
        else:
            min_ts = 0.0

    if (
        getattr(cfg, "run_mode", None) == "teacher_target_distill"
        or getattr(cfg, "student_attn_impl", None) == "wda"
    ):
        min_ts = 0.0

    # EMA stage: optional floor on teacher_scale to avoid endpoint cliffs.
    if bool(getattr(cfg, "ema_teacher_enabled", False)) and step >= float(
        getattr(cfg, "ema_start_progress", 0.0)
    ):
        floor_start = float(getattr(cfg, "ema_teacher_scale_floor", 0.05))
        floor_end = getattr(cfg, "ema_teacher_scale_floor_end", None)
        if floor_end is None:
            floor_end = floor_start
        floor_end = float(floor_end)
        floor_steps = int(getattr(cfg, "ema_teacher_scale_floor_steps", 0))
        p_floor = _ramp(
            step,
            start_step=int(getattr(cfg, "ema_start_progress", 0.0)),
            steps=floor_steps,
        )
        floor = floor_start + p_floor * (floor_end - floor_start)
        min_ts = max(min_ts, floor)

    modules_iter = wrappers if wrappers is not None else model.modules()

    if cfg.handoff_mode == "sequential":
        for m in modules_iter:
            if wrappers is None and not isinstance(m, ParallelAttentionWrapper):
                continue
            l_idx = m.layer_idx
            if l_idx is None:
                continue

            l_period = cfg.steps_per_layer + cfg.handoff_stabilization_steps
            l_start = cfg.handoff_start_step + l_idx * l_period
            l_end = l_start + cfg.steps_per_layer

            if step < l_start:
                ts = cfg.teacher_scale_start
            elif step >= l_end:
                ts = cfg.teacher_scale_end
            else:
                p = (step - l_start) / float(cfg.steps_per_layer)
                ts = (1.0 - p) ** cfg.handoff_power
                ts = cfg.teacher_scale_end + ts * (
                    cfg.teacher_scale_start - cfg.teacher_scale_end
                )

            ts = max(ts, min_ts)

            # WDA scale usually stays at 1.0 or ramps up
            p_global = _ramp(
                step, start_step=cfg.handoff_start_step, steps=cfg.handoff_steps
            )
            ws = cfg.wda_scale_start + p_global * (
                cfg.wda_scale_end - cfg.wda_scale_start
            )

            m.set_alpha(alpha)
            m.set_scales(teacher_scale=ts, wda_scale=ws)
            layer_teacher_scales[l_idx] = ts
    else:
        # Global handoff
        p = _ramp(step, start_step=cfg.handoff_start_step, steps=cfg.handoff_steps)
        teacher_scale = float(
            cfg.teacher_scale_start
            + p * (cfg.teacher_scale_end - cfg.teacher_scale_start)
        )
        teacher_scale = max(teacher_scale, min_ts)

        wda_scale = float(
            cfg.wda_scale_start + p * (cfg.wda_scale_end - cfg.wda_scale_start)
        )
        for m in modules_iter:
            if wrappers is None and not isinstance(m, ParallelAttentionWrapper):
                continue
            m.set_alpha(alpha)
            m.set_scales(teacher_scale=teacher_scale, wda_scale=wda_scale)
            if m.layer_idx is not None:
                layer_teacher_scales[m.layer_idx] = teacher_scale

    return layer_teacher_scales


def _update_migration_state(
    optimizer: torch.optim.Optimizer, step: int, cfg: TransplantConfig, base_lr: float
):
    """Updates parameter group LRs and frozen states for sequential migration."""
    if cfg.handoff_mode != "sequential":
        # Global handoff fallback
        for group in optimizer.param_groups:
            group["lr"] = base_lr
        return

    l_period = cfg.steps_per_layer + cfg.handoff_stabilization_steps

    for group in optimizer.param_groups:
        l_idx = group.get("layer_idx", -1)
        if l_idx == -1:
            group["lr"] = base_lr
            continue

        # Option: restrict training to specific layers (Surgical Focus)
        if cfg.active_layers is not None:
            if l_idx not in cfg.active_layers:
                group["lr"] = 0.0
                for p in group["params"]:
                    p.requires_grad = False
                continue
            else:
                for p in group["params"]:
                    p.requires_grad = True

        l_start = cfg.handoff_start_step + l_idx * l_period
        l_end = l_start + cfg.steps_per_layer

        # A layer is 'converted' if it has finished its handoff decay window.
        is_converted = step >= l_end

        if is_converted:
            if cfg.freeze_converted:
                # Hard freeze
                for p in group["params"]:
                    p.requires_grad = False
                group["lr"] = 0.0
            else:
                # Soft freeze: reduce learning rate
                group["lr"] = base_lr * cfg.converted_lr_scale
        else:
            # Active or future layer: full learning rate
            group["lr"] = base_lr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transplant_gemma270m.json")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--resume_adapter", default=None, help="Path to adapter_step*.pt to resume from"
    )
    ap.add_argument(
        "--patch_layers",
        default="all",
        help="Which decoder layer indices to patch (e.g. 'all' or '0' or '0,1,2')",
    )
    ap.add_argument(
        "--override_progress",
        type=float,
        default=None,
        help="Manually set the initial autopilot progress (e.g. to recover lost prog)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    cfg = apply_feature_flags(cfg)
    schedule_manager = ScheduleManager(cfg.schedule_stages)

    teacher_free_mode = bool(getattr(cfg, "teacher_free_mode", False))

    device = str(cfg.device)
    dtype = parse_torch_dtype(cfg.torch_dtype)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"Teacher: {cfg.teacher_model_id}")

    if hasattr(cfg, "datasets") and cfg.datasets:
        print(f"Dataset: Mixed mixture of {len(cfg.datasets)} sources")
    else:
        print(f"Dataset: {cfg.dataset_path}")

    print(
        f"Seq len: {cfg.seq_len} | micro_bs={cfg.micro_batch_size} | grad_accum={cfg.grad_accum}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = getattr(cfg, "quantization", None)
    # Optional: skip loading external teacher if we start directly in EMA-teacher mode.
    ema_enabled = bool(getattr(cfg, "ema_teacher_enabled", False)) and (
        not teacher_free_mode
    )
    ema_start_progress = float(getattr(cfg, "ema_start_progress", 0.0))

    # If we are resuming and will immediately be in EMA-teacher mode, we can avoid
    # loading the external teacher model entirely (saves VRAM/time on Phase-D runs).
    resume_path_hint = args.resume_adapter
    if resume_path_hint is None:
        try:
            out_dir_hint = Path(cfg.out_dir)
            ckpts = sorted(
                out_dir_hint.glob("adapter_step*.pt"),
                key=lambda x: (
                    int(re.search(r"step(\d+)", x.name).group(1))
                    if re.search(r"step(\d+)", x.name)
                    else 0
                ),
            )
            if ckpts:
                resume_path_hint = str(ckpts[-1])
        except Exception:
            resume_path_hint = None

    resume_hint_progress = args.override_progress
    if resume_hint_progress is None and resume_path_hint is not None and ema_enabled:
        try:
            ckpt = torch.load(resume_path_hint, map_location="cpu")
            if isinstance(ckpt, dict):
                meta = (
                    ckpt.get("payload", {})
                    if isinstance(ckpt.get("payload", None), dict)
                    else {}
                )
                if "autopilot_progress" in meta:
                    resume_hint_progress = float(meta["autopilot_progress"])
                elif "auto_progress" in meta:
                    resume_hint_progress = float(meta["auto_progress"])
        except Exception:
            resume_hint_progress = None

    teacher = None
    if teacher_free_mode:
        print("Teacher-free burn-in mode: skipping external teacher load (CE-only).")

    skip_external_teacher = False
    if (
        (not teacher_free_mode)
        and ema_enabled
        and resume_hint_progress is not None
        and resume_hint_progress >= ema_start_progress
    ):
        skip_external_teacher = True
        print(
            f"Resume indicates EMA mode active (progress≈{resume_hint_progress:.1f} >= {ema_start_progress:.1f}); "
            "skipping external teacher load."
        )

    if (not teacher_free_mode) and not (
        ema_enabled and (ema_start_progress <= 0.0 or skip_external_teacher)
    ):
        teacher = _load_causal_lm(
            cfg.teacher_model_id, dtype=dtype, device=device, quantization=quant
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    else:
        if (not teacher_free_mode) and ema_enabled and ema_start_progress <= 0.0:
            print(
                "EMA-teacher enabled with ema_start_progress<=0: skipping external teacher load."
            )

    student = _load_causal_lm(
        cfg.teacher_model_id, dtype=dtype, device=device, quantization=quant
    )

    # Teacher-free runs may ramp teacher_scale down to 0 within a single run.
    # If gammas are preserved during the early (anchored) part of the run, we still
    # want an automatic one-time stabilization as we approach true WDA-only.
    tf_gamma_adjusted_once = False

    if cfg.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    # Freeze everything first.
    for p in student.parameters():
        p.requires_grad = False

    hidden_size = int(
        getattr(student.config, "hidden_size", getattr(student.config, "n_embd", None))
    )
    if hidden_size is None:
        raise ValueError("Could not infer hidden_size from model.config")
    num_heads = _get_num_heads(student.config)
    num_layers = _get_num_layers(student.config)

    if str(args.patch_layers).lower() == "all":
        patch_layer_set: set[int] | None = None
    else:
        patch_layer_set = set()
        for part in str(args.patch_layers).split(","):
            part = part.strip()
            if not part:
                continue
            patch_layer_set.add(int(part))

    def make_wrapper(attn_mod: torch.nn.Module, path: str) -> torch.nn.Module:
        layer_idx = _infer_layer_index(path)
        if layer_idx is None:
            layer_scale = 1.0
        else:
            layer_scale = _layer_alpha_scale(
                layer_idx, num_layers, cfg.alpha_layer_scale
            )
        wda = WDABridge(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=cfg.seq_len,
            num_masks=cfg.wda_num_masks,
            num_waves_per_mask=cfg.wda_num_waves_per_mask,
            topk_masks=cfg.wda_topk_masks,
            gate_temp=getattr(cfg, "wda_gate_temp", 1.0),
            attn_alpha=cfg.wda_attn_alpha,
            content_mix=cfg.wda_content_mix,
            learned_content=cfg.wda_learned_content,
            use_sin_waves=cfg.wda_use_sin_waves,
            use_sampling=cfg.wda_use_sampling,
            num_samples=cfg.wda_num_samples,
            noise_sigma=cfg.wda_noise_sigma,
            step_alpha=cfg.wda_step_alpha,
            use_checkpoint=cfg.wda_use_checkpoint,
        )
        return ParallelAttentionWrapper(
            attn_mod,
            wda,
            layer_idx=layer_idx,
            init_alpha=0.0,
            init_teacher_scale=1.0,
            init_wda_scale=1.0,
            init_layer_alpha_scale=layer_scale,
            init_gamma=cfg.init_gamma,
        )

    def filter_paths(path: str) -> bool:
        if patch_layer_set is None:
            return True
        idx = _infer_layer_index(path)
        return idx is not None and idx in patch_layer_set

    patched_paths = patch_attention_modules(
        student, make_wrapper, filter_paths=filter_paths
    )
    print(f"Patched {len(patched_paths)} attention modules")

    wrappers = [m for m in student.modules() if isinstance(m, ParallelAttentionWrapper)]

    distill_enabled = bool(getattr(cfg, "distill_enabled", False))
    distill_targets = cfg.distill_targets or []
    distill_needs_hidden = distill_enabled and (
        "hidden_states" in distill_targets or "resid_energy" in distill_targets
    )
    feature_extractor = FeatureExtractor(distill_targets)
    ParallelAttentionWrapper.collect_features = bool(
        distill_enabled and feature_extractor.wants_wrapper_features()
    )

    # Selection logic: Unfreeze either gamma/gain only, or full trainable set.
    layer_params = {}  # {idx: [params]}
    other_params = []

    train_gamma_only = bool(getattr(cfg, "train_gamma_only", False))
    for name, p in student.named_parameters():
        # NEVER unfreeze the teacher reference branch inside the wrapper
        if ".teacher_attn." in name:
            p.requires_grad = False
            continue

        if train_gamma_only:
            # We train gamma/gain, AND the student attention weights (WDA)
            # but we keep the backbone (MLP, LN) frozen.
            p.requires_grad = (
                name.endswith(".gamma")
                or name.endswith(".wda_gain")
                or ".wda_attn." in name
            )
        else:
            # Unfreeze WDA, Gammas, AND the backbone (MLPs, LayerNorms, Embeddings, Head)
            p.requires_grad = True

        if not p.requires_grad:
            continue

        l_idx = _infer_layer_index(name)
        if l_idx is not None:
            if l_idx not in layer_params:
                layer_params[l_idx] = []
            layer_params[l_idx].append(p)
        else:
            other_params.append(p)

    param_groups = []
    # Add layers in order, separating scalars from weights if gamma_only_lr is set
    gamma_only_lr = getattr(cfg, "gamma_only_lr", None)
    gamma_lr = float(gamma_only_lr) if (train_gamma_only and gamma_only_lr) else cfg.lr

    def _get_group_lr(params_list):
        # If any param in this list is NOT a scalar (gamma/gain), use base lr
        # Actually, let's just make two groups per layer if needed.
        scalar_params = []
        weight_params = []
        for p in params_list:
            # We can't easily check name here, but we can check shape.
            # Gamma and Gain are scalars (numel == 1).
            if p.numel() == 1:
                scalar_params.append(p)
            else:
                weight_params.append(p)
        return scalar_params, weight_params

    for l_idx in sorted(layer_params.keys()):
        s_p, w_p = _get_group_lr(layer_params[l_idx])
        if s_p:
            param_groups.append({"params": s_p, "layer_idx": l_idx, "lr": gamma_lr})
        if w_p:
            param_groups.append({"params": w_p, "layer_idx": l_idx, "lr": cfg.lr})

    if other_params:
        s_p, w_p = _get_group_lr(other_params)
        if s_p:
            param_groups.append({"params": s_p, "layer_idx": -1, "lr": gamma_lr})
        if w_p:
            param_groups.append({"params": w_p, "layer_idx": -1, "lr": cfg.lr})

    if cfg.gradient_checkpointing:
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        else:
            emb = getattr(student, "get_input_embeddings", lambda: None)()
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad = True

    num_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"Trainable params: {num_trainable/1e6:.2f}M")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume Logic
    start_step = 1
    resume_path = args.resume_adapter

    # Resume-specific autopilot state
    resume_auto_progress = None
    resume_auto_speed = None

    if resume_path is None:
        # Check for existing checkpoints to auto-resume
        ckpts = sorted(
            out_dir.glob("adapter_step*.pt"),
            key=lambda x: (
                int(re.search(r"step(\d+)", x.name).group(1))
                if re.search(r"step(\d+)", x.name)
                else 0
            ),
        )
        if ckpts:
            resume_path = str(ckpts[-1])
            print(f"Auto-resuming from latest checkpoint: {resume_path}")

    if resume_path is not None:
        info = load_adapter(student, resume_path, strict=False)
        payload = info.get("payload", {})
        if isinstance(payload, dict):
            # Our checkpoints store user-supplied metadata in the nested "payload" dict.
            # Older/alternate formats may place fields at the top-level, so we support both.
            meta = (
                payload.get("payload", {})
                if isinstance(payload.get("payload", None), dict)
                else {}
            )

            if "step" in payload and payload["step"] is not None:
                start_step = int(payload["step"]) + 1
                print(
                    f"Resumed from step {payload['step']}, starting at step {start_step}"
                )
            elif "step" in meta and meta["step"] is not None:
                start_step = int(meta["step"]) + 1
                print(
                    f"Resumed from step {meta['step']}, starting at step {start_step}"
                )

            # Check for saved autopilot state (nested-first, top-level fallback)
            if "autopilot_progress" in meta:
                resume_auto_progress = float(meta["autopilot_progress"])
                print(f"Loaded autopilot progress: {resume_auto_progress:.1f}")
            elif "autopilot_progress" in payload:
                resume_auto_progress = float(payload["autopilot_progress"])
                print(f"Loaded autopilot progress: {resume_auto_progress:.1f}")

            if "autopilot_speed" in meta:
                resume_auto_speed = float(meta["autopilot_speed"])
                print(f"Loaded autopilot speed: {resume_auto_speed:.3f}")
            elif "autopilot_speed" in payload:
                resume_auto_speed = float(payload["autopilot_speed"])
                print(f"Loaded autopilot speed: {resume_auto_speed:.3f}")

        if start_step == 1:
            # Try to infer step from filename
            m = re.search(r"step(\d+)", Path(resume_path).name)
            if m:
                start_step = int(m.group(1)) + 1
                print(f"Inferred start step from filename: {start_step}")

        print(
            f"Resumed adapter: loaded={info.get('loaded', 0)} skipped={info.get('skipped', 0)} shape_mismatch={info.get('shape_mismatch', 0)}"
        )

    # Teacher-free stabilization: ensure gammas are sane before training.
    if teacher_free_mode:
        tf_ts = float(getattr(cfg, "teacher_free_teacher_scale", 0.0))
        gamma_reset = getattr(cfg, "teacher_free_gamma_reset", None)

        # AUTO default: only reset gammas when we are truly WDA-only.
        # This preserves landing-style runs where a small teacher_scale anchor is kept.
        if gamma_reset is None:
            do_gamma_reset = tf_ts <= 1e-6
        else:
            do_gamma_reset = bool(gamma_reset)

        if do_gamma_reset:
            gamma_init = getattr(cfg, "teacher_free_gamma_init", 1.0)
            gamma_min = float(getattr(cfg, "teacher_free_gamma_min", 0.05))
            did_gamma_change = (gamma_init is not None) or (gamma_min > 0.0)
            if did_gamma_change:
                with torch.no_grad():
                    for m in student.modules():
                        if isinstance(m, ParallelAttentionWrapper):
                            if gamma_init is not None:
                                m.gamma.fill_(float(gamma_init))
                            if gamma_min > 0.0:
                                # Clamp (don't abs-flip) to avoid accidental sign changes.
                                m.gamma.clamp_(min=gamma_min)
                tf_gamma_adjusted_once = True
                print(
                    f"Teacher-free mode: adjusted wrapper gammas (init={gamma_init}, min={gamma_min}, ts={tf_ts})."
                )
        else:
            print(
                f"Teacher-free mode: preserving wrapper gammas (gamma_reset={gamma_reset}, ts={tf_ts})."
            )

    # If param_groups already have 'lr', this serves as the default for any that don't.
    use_fused_adamw = bool(getattr(cfg, "use_fused_adamw", False))
    opt = None
    if use_fused_adamw:
        if not str(device).startswith("cuda"):
            print(
                "Fused AdamW requested but CUDA is not available; using standard AdamW."
            )
        else:
            try:
                sig = inspect.signature(torch.optim.AdamW)
                if "fused" in sig.parameters:
                    opt = torch.optim.AdamW(
                        param_groups,
                        lr=cfg.lr,
                        weight_decay=cfg.weight_decay,
                        fused=True,
                    )
                    print("Using fused AdamW optimizer.")
                else:
                    print(
                        "Fused AdamW not supported in this PyTorch build; using standard AdamW."
                    )
            except Exception as exc:
                print(f"Fused AdamW unavailable ({exc}); using standard AdamW.")

    if opt is None:
        opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    static_trainable = (
        cfg.handoff_mode != "sequential"
        and cfg.active_layers is None
        and (not cfg.freeze_converted)
    )
    all_trainable = None
    if static_trainable:
        all_trainable = [
            p for g in opt.param_groups for p in g["params"] if p.requires_grad
        ]

    # ---- EMA Teacher State (Mean Teacher) ----
    # We store EMA weights only for trainable parameters (requires_grad=True).
    ema_params: dict[str, torch.Tensor] | None = None
    ema_model_params: dict[str, torch.Tensor] | None = None
    ema_lm_params: dict[str, torch.Tensor] | None = None
    teacher_offloaded = False
    if ema_enabled:
        if _functional_call is None:
            raise RuntimeError(
                "EMA teacher requested but torch.nn.utils.stateless.functional_call is unavailable. "
                "Upgrade PyTorch to 2.x or disable ema_teacher_enabled."
            )
        ema_decay = float(getattr(cfg, "ema_decay", 0.9995))
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0,1), got {ema_decay}")

        # Initialize EMA from current student weights.
        ema_params = {}
        for name, p in student.named_parameters():
            if p.requires_grad:
                ema_params[name] = p.detach().clone()

        # Pre-build views for functional_call on submodules.
        ema_model_params = {
            k[len("model.") :]: v
            for k, v in ema_params.items()
            if k.startswith("model.")
        }
        ema_lm_params = {
            k[len("lm_head.") :]: v
            for k, v in ema_params.items()
            if k.startswith("lm_head.")
        }

        print(
            f"EMA teacher enabled: decay={ema_decay} start_progress={ema_start_progress} "
            f"tracked_params={len(ema_params)}"
        )

    teacher_provider = TeacherProvider(
        student=student,
        teacher=teacher,
        ema_model_params=ema_model_params,
        ema_lm_params=ema_lm_params,
        functional_call=_functional_call,
    )

    # ... lines 435-445 ...
    # RESUME WARMUP: if we are not at step 1, we need to re-warmup the optimizer LR
    # to avoid divergence from zero momenta.
    resumed_from_ckpt = start_step > 1

    # Use get_data_streamer to support mixed datasets and loss masking.
    from wave_dencity.transplant.data import get_data_streamer

    # Handle both old 'dataset_path' string and new 'datasets' list in config.
    dataset_cfg = (
        cfg.datasets
        if (hasattr(cfg, "datasets") and cfg.datasets is not None)
        else cfg.dataset_path
    )

    stream = get_data_streamer(
        tokenizer,
        dataset_cfg=dataset_cfg,
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        device=device,
    )

    # Sanity check: alpha=0 student == teacher
    # Only meaningful on fresh runs (no resumed adapter), since we train the backbone.
    if teacher is not None and start_step == 1:
        print("Running alpha=0 sanity check (eval mode)...")
        student.eval()
        # Dummy step 0 for sanity check
        _set_scales(student, step=0, cfg=cfg, num_layers=num_layers, alpha=0.0)
        batch = next(stream)
        with torch.no_grad():
            t_logits = teacher(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits
            s_logits = student(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits
            max_diff = (t_logits - s_logits).abs().max().item()
        print(f"alpha=0 max|diff_logits| = {max_diff:.6f}")
        student.train()
    else:
        print("Skipping alpha=0 sanity check")

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.startswith("cuda") and dtype in {torch.float16}
    )
    amp_enabled = device.startswith("cuda")

    t0 = time.time()
    tokens_seen = 0

    log_path = (
        Path(cfg.log_jsonl_path) if cfg.log_jsonl_path else Path("training_log.jsonl")
    )
    if not log_path.is_absolute():
        if not (
            len(log_path.parts) >= len(out_dir.parts)
            and tuple(log_path.parts[: len(out_dir.parts)]) == tuple(out_dir.parts)
        ):
            log_path = out_dir / log_path
    sqlite_path = None
    if cfg.log_sqlite_path:
        sqlite_path = Path(cfg.log_sqlite_path)
        if not sqlite_path.is_absolute():
            if not (
                len(sqlite_path.parts) >= len(out_dir.parts)
                and tuple(sqlite_path.parts[: len(out_dir.parts)])
                == tuple(out_dir.parts)
            ):
                sqlite_path = out_dir / sqlite_path
    logger = StepLogger(
        str(log_path),
        include=cfg.log_fields_include,
        exclude=cfg.log_fields_exclude,
        sqlite_path=str(sqlite_path) if sqlite_path else None,
        sqlite_table=cfg.log_sqlite_table,
        sqlite_every=cfg.log_sqlite_every,
    )

    resumed_warmup_steps = 10

    # Yellow Zone tracking
    yellow_zone_active = 0  # Steps remaining in cooldown
    staged_kl_boost = 0.0
    staged_ce_reduction = 0.0
    staged_gt_softness = 0.0

    autopilot = None
    if cfg.autopilot_enabled:
        # Use overridden progress, then resumed progress, then starting step
        start_p = args.override_progress
        if start_p is None:
            start_p = (
                resume_auto_progress
                if resume_auto_progress is not None
                else float(start_step)
            )

        autopilot = Autopilot(
            target_low=cfg.autopilot_kl_low,
            target_high=cfg.autopilot_kl_high,
            start_progress=start_p,
        )
        if resume_auto_speed is not None and args.override_progress is None:
            autopilot.speed = resume_auto_speed
            print(f"Restored autopilot speed: {autopilot.speed:.3f}")

        if args.override_progress is not None:
            print(f"!!! MANUAL PROGRESS OVERRIDE: Starting at {start_p}")

    step = start_step
    target_step = None
    run_steps = int(getattr(cfg, "run_steps", 0) or 0)
    if run_steps > 0:
        target_step = int(start_step) + int(run_steps)
        print(f"Run-limited mode: run_steps={run_steps} -> target_step={target_step}")
    while True:
        # Calculate progress (uses autopilot if enabled)
        current_progress = autopilot.progress if autopilot else float(step)

        use_ema_teacher = ema_enabled and (current_progress >= ema_start_progress)
        teacher_provider.set_use_ema(use_ema_teacher)

        # If we switched into EMA teacher mode, offload external teacher ASAP.
        if use_ema_teacher and (teacher is not None) and not teacher_offloaded:
            print(
                ">>> EMA TEACHER: Offloading external teacher model to CPU and freeing VRAM"
            )
            try:
                teacher.to("cpu")
            except Exception:
                pass
            teacher = None
            teacher_provider.set_teacher(teacher)
            teacher_offloaded = True
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Stop condition
        if target_step is not None:
            if step >= target_step:
                print(f"Goal reached! (Step: {step} / Target step: {target_step})")
                break
        else:
            # Break if we've reached the target progress (or target steps if no autopilot)
            if current_progress >= cfg.steps:
                print(
                    f"Goal reached! (Progress: {current_progress:.2f} / Target: {cfg.steps})"
                )
                break

        # Calculate Base LR with resume ramp if needed
        base_lr = cfg.lr
        if resumed_from_ckpt and (step < start_step + resumed_warmup_steps):
            local_step = step - start_step + 1
            ramp = min(1.0, local_step / float(resumed_warmup_steps))
            base_lr = cfg.lr * ramp

        # Tie LR to autopilot speed if enabled (reduces learning rate when struggling)
        if autopilot:
            # Floor LR reduction at 10% of base to ensure we can still learn out of a bad state
            lr_factor = max(0.1, autopilot.speed)
            base_lr = base_lr * lr_factor

        # --- CURRICULUM SCHEDULING (PHASE B1 / B2 / C / EMA) ---

        # 1. Base Values
        base_gate_temp = 0.2 + _ramp(
            current_progress, start_step=cfg.handoff_start_step, steps=cfg.handoff_steps
        ) * (cfg.wda_gate_temp - 0.2)
        base_alpha = _alpha_for_step(
            current_progress,
            alpha_max=cfg.alpha_max,
            warmup_steps=cfg.alpha_warmup_steps,
        )
        cur_kl_w = cfg.kl_weight
        cur_ce_w = cfg.ce_weight
        cur_temp = float(cfg.temperature)

        if teacher_free_mode:
            # Teacher-free burn-in: avoid schedule discontinuities, and do CE-only.
            alpha = float(getattr(cfg, "teacher_free_alpha", 1.0))
            current_gate_temp = float(getattr(cfg, "wda_gate_temp", 0.05))
            cur_kl_w = 0.0
            cur_ce_w = float(getattr(cfg, "ce_weight", 1.0))
            cur_temp = float(getattr(cfg, "temperature", 1.0))
        else:
            # 2. Phase B1: Autonomy Stabilization (Safe Harbor, 2000-3500)
            if current_progress >= 2000 and current_progress <= 3500:
                alpha = min(base_alpha, 0.85)
                current_gate_temp = max(base_gate_temp, 0.10)
                cur_kl_w = 0.6
                cur_ce_w = 1.0
                cur_temp = 1.1

            # 3. Phase B2/C: Specialization & Final Release (3500-5500)
            elif current_progress > 3500:
                p_glide = _ramp(current_progress, start_step=3500, steps=2000)
                alpha = 0.85 + p_glide * (min(1.0, cfg.alpha_max) - 0.85)
                current_gate_temp = 0.10 + p_glide * (cfg.wda_gate_temp - 0.10)
                cur_kl_w = 0.6 + p_glide * (0.3 - 0.6)
                cur_ce_w = 1.0 + p_glide * (1.5 - 1.0)
                cur_temp = 1.1 + p_glide * (1.0 - 1.1)

            else:
                # Phase A: Initial handoff ramp
                alpha = base_alpha
                current_gate_temp = base_gate_temp
                # Use original config weights or B1 ramp
                p_b1_start = _ramp(current_progress, start_step=1600, steps=400)
                cur_kl_w = cfg.kl_weight + p_b1_start * (0.6 - cfg.kl_weight)
                cur_ce_w = cfg.ce_weight + p_b1_start * (1.0 - cfg.ce_weight)

            # 5. EMA Stage Override (avoids endpoint cliffs)
            if use_ema_teacher:
                ema_alpha_cap_start = float(getattr(cfg, "ema_alpha_cap", 0.995))
                ema_alpha_cap_end = getattr(cfg, "ema_alpha_cap_end", None)
                if ema_alpha_cap_end is None:
                    ema_alpha_cap_end = ema_alpha_cap_start
                ema_alpha_cap_end = float(ema_alpha_cap_end)
                ema_alpha_cap_steps = int(getattr(cfg, "ema_alpha_cap_steps", 0))

                p_cap = _ramp(
                    current_progress,
                    start_step=int(ema_start_progress),
                    steps=ema_alpha_cap_steps,
                )
                ema_alpha_cap = ema_alpha_cap_start + p_cap * (
                    ema_alpha_cap_end - ema_alpha_cap_start
                )
                alpha = min(alpha, ema_alpha_cap)

                # Gate temp: hold softer routing during EMA stage, then taper
                ema_gt_start = float(getattr(cfg, "ema_gate_temp_start", 0.08))
                ema_gt_end = float(getattr(cfg, "ema_gate_temp_end", 0.05))
                ema_gt_steps = int(getattr(cfg, "ema_gate_temp_steps", 1000))
                p_ema = _ramp(
                    current_progress,
                    start_step=int(ema_start_progress),
                    steps=ema_gt_steps,
                )
                current_gate_temp = ema_gt_start + p_ema * (ema_gt_end - ema_gt_start)

                # Loss weights
                cur_kl_w = float(getattr(cfg, "ema_kl_weight", 0.2))
                cur_ce_w = float(getattr(cfg, "ema_ce_weight", 1.5))

        schedule_overrides = (
            schedule_manager.get_overrides(current_progress)
            if schedule_manager is not None
            else None
        )
        teacher_scale_override = None
        wda_scale_override = None
        tf_teacher_scale_override = None
        tf_wda_scale_override = None
        lr_override = None
        lr_scale_override = None
        if schedule_overrides:
            if "alpha" in schedule_overrides:
                alpha = float(schedule_overrides["alpha"])
            if "gate_temp" in schedule_overrides:
                current_gate_temp = float(schedule_overrides["gate_temp"])
            if "kl_weight" in schedule_overrides:
                cur_kl_w = float(schedule_overrides["kl_weight"])
            if "ce_weight" in schedule_overrides:
                cur_ce_w = float(schedule_overrides["ce_weight"])
            if "temperature" in schedule_overrides:
                cur_temp = float(schedule_overrides["temperature"])
            if "teacher_scale" in schedule_overrides:
                teacher_scale_override = float(schedule_overrides["teacher_scale"])
            if "wda_scale" in schedule_overrides:
                wda_scale_override = float(schedule_overrides["wda_scale"])
            if "teacher_free_teacher_scale" in schedule_overrides:
                tf_teacher_scale_override = float(
                    schedule_overrides["teacher_free_teacher_scale"]
                )
            if "teacher_free_wda_scale" in schedule_overrides:
                tf_wda_scale_override = float(
                    schedule_overrides["teacher_free_wda_scale"]
                )
            if "lr" in schedule_overrides:
                lr_override = float(schedule_overrides["lr"])
            if "lr_scale" in schedule_overrides:
                lr_scale_override = float(schedule_overrides["lr_scale"])

        # 4. Yellow Zone Safeguards (Temporary overrides for stability)
        if yellow_zone_active > 0:
            current_gate_temp = max(current_gate_temp, 0.10 + staged_gt_softness)
            cur_kl_w = min(1.0, cur_kl_w + staged_kl_boost)
            cur_ce_w = max(0.5, cur_ce_w - staged_ce_reduction)
            yellow_zone_active -= 1

        # Apply scheduled LR overrides after other adjustments.
        if lr_override is not None:
            base_lr = float(max(1e-12, lr_override))
        if lr_scale_override is not None:
            base_lr = float(max(1e-12, base_lr * lr_scale_override))

        # Apply settings back to model
        for m in wrappers:
            if hasattr(m.wda_attn, "gate_temp"):
                m.wda_attn.gate_temp = current_gate_temp

        if teacher_free_mode:
            tf_ts_start = float(getattr(cfg, "teacher_free_teacher_scale", 0.0))
            tf_ts_end = getattr(cfg, "teacher_free_teacher_scale_end", None)
            tf_ts_steps = int(getattr(cfg, "teacher_free_teacher_scale_steps", 0) or 0)

            tf_teacher_scale = tf_ts_start
            if tf_ts_end is not None and tf_ts_steps > 0:
                try:
                    # local optimizer step offset since this run started
                    local_step = int(step) - int(start_step)
                except Exception:
                    local_step = 0
                p_ts = max(0.0, min(1.0, local_step / float(tf_ts_steps)))
                tf_teacher_scale = tf_ts_start + p_ts * (float(tf_ts_end) - tf_ts_start)
            if tf_teacher_scale_override is not None:
                tf_teacher_scale = tf_teacher_scale_override

            # Auto gamma stabilization for ramp-to-zero runs.
            # If we preserved checkpoint gammas during the anchored portion of the run,
            # we still want a stable, non-degenerate gamma regime before ts reaches 0.
            #
            # Semantics:
            # - teacher_free_gamma_reset=True  -> handled at startup (always)
            # - teacher_free_gamma_reset=False -> never touch gammas
            # - teacher_free_gamma_reset=None  -> auto: apply once as we approach WDA-only
            gamma_reset = getattr(cfg, "teacher_free_gamma_reset", None)
            is_ramping_to_zero = (
                tf_ts_end is not None
                and tf_ts_steps > 0
                and float(tf_ts_end) <= 1e-6
                and tf_ts_start > 0.0
            )
            if (
                (not tf_gamma_adjusted_once)
                and (gamma_reset is None)
                and is_ramping_to_zero
            ):
                # Trigger early enough to give the model a handful of steps to adapt
                # *before* ts hits 0. (With a linear ts ramp, ts<=1e-3 is ~last 90 steps
                # when starting from 0.02 over 1800 steps.)
                if tf_teacher_scale <= 1e-3:
                    gamma_init = getattr(cfg, "teacher_free_gamma_init", 1.0)
                    gamma_min = float(getattr(cfg, "teacher_free_gamma_min", 0.05))
                    did_gamma_change = (gamma_init is not None) or (gamma_min > 0.0)
                    if did_gamma_change:
                        with torch.no_grad():
                            for m in wrappers:
                                if gamma_init is not None:
                                    m.gamma.fill_(float(gamma_init))
                                if gamma_min > 0.0:
                                    m.gamma.clamp_(min=gamma_min)
                        tf_gamma_adjusted_once = True
                        print(
                            "Teacher-free auto: stabilized gammas while ramping teacher_scale->0 "
                            f"(init={gamma_init}, min={gamma_min}, ts≈{tf_teacher_scale:.3g})."
                        )

            tf_ws_start = float(getattr(cfg, "teacher_free_wda_scale", 1.0))
            tf_ws_end = getattr(cfg, "teacher_free_wda_scale_end", None)
            tf_ws_steps = int(getattr(cfg, "teacher_free_wda_scale_steps", 0) or 0)
            tf_wda_scale = tf_ws_start
            if tf_ws_end is not None and tf_ws_steps > 0:
                try:
                    local_step = int(step) - int(start_step)
                except Exception:
                    local_step = 0
                p_ws = max(0.0, min(1.0, local_step / float(tf_ws_steps)))
                tf_wda_scale = tf_ws_start + p_ws * (float(tf_ws_end) - tf_ws_start)
            if tf_wda_scale_override is not None:
                tf_wda_scale = tf_wda_scale_override
            teacher_scales = {}
            for m in wrappers:
                m.set_alpha(alpha)
                m.set_scales(teacher_scale=tf_teacher_scale, wda_scale=tf_wda_scale)
                if m.layer_idx is not None:
                    teacher_scales[m.layer_idx] = tf_teacher_scale
        else:
            teacher_scales = _set_scales(
                student,
                step=current_progress,
                cfg=cfg,
                num_layers=num_layers,
                alpha=alpha,
                wrappers=wrappers,
            )
            if teacher_scale_override is not None or wda_scale_override is not None:
                ts_val = teacher_scale_override
                ws_val = wda_scale_override
                for m in wrappers:
                    m.set_scales(
                        teacher_scale=ts_val if ts_val is not None else None,
                        wda_scale=ws_val if ws_val is not None else None,
                    )
                if ts_val is not None:
                    teacher_scales = {
                        m.layer_idx: float(ts_val)
                        for m in wrappers
                        if m.layer_idx is not None
                    }
        _update_migration_state(opt, current_progress, cfg, base_lr)

        collect_stats = (step % cfg.log_every == 0) or (step == 1)
        ParallelAttentionWrapper.collect_stats = collect_stats

        opt.zero_grad(set_to_none=True)
        total_loss, total_kl, total_ce, total_h = 0.0, 0.0, 0.0, 0.0
        total_feature = 0.0
        feature_breakdown_totals: dict[str, float] = {}
        bad_step = False

        for _ in range(cfg.grad_accum):
            batch = next(stream)
            kl, ce = 0.0, 0.0
            tokens_seen += int(batch.attention_mask.sum().item())

            need_teacher_targets = (not teacher_free_mode) and (
                (cur_kl_w > 0.0) or (cfg.hidden_match_weight > 0.0) or distill_enabled
            )

            t_hidden = None
            t_all_h = None
            if need_teacher_targets:
                with torch.no_grad():
                    t_hidden, t_all_h = teacher_provider.forward_backbone(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        output_hidden_states=(cfg.hidden_match_weight > 0.0)
                        or distill_needs_hidden,
                    )

            with torch.amp.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                enabled=amp_enabled,
                dtype=dtype,
            ):
                # Use current scheduled temperature
                T = cur_temp

                # Get student backbone hidden states
                s_out = student.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    output_hidden_states=(cfg.hidden_match_weight > 0.0)
                    or distill_needs_hidden,
                )
                s_hidden = s_out[0]
                s_all_h = (
                    s_out.hidden_states
                    if (cfg.hidden_match_weight > 0.0) or distill_needs_hidden
                    else None
                )

                internal_h_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                if (
                    cfg.hidden_match_weight > 0.0
                    and t_all_h is not None
                    and s_all_h is not None
                ):
                    # Compare layers. Note: hidden_states has num_layers + 1 (entry embedding + each layer)
                    for i in range(1, len(t_all_h)):
                        l_idx = i - 1
                        # Weight early layers higher if they are active
                        l_weight = 1.0
                        if cfg.active_layers is not None and l_idx in cfg.active_layers:
                            if l_idx < 4:  # Early layers bonus
                                l_weight = 5.0
                            else:
                                l_weight = 2.0

                        # MSE on hidden states, masked by attention_mask
                        diff = (s_all_h[i] - t_all_h[i]).pow(2)
                        # Mean over masked tokens
                        mask = batch.attention_mask.unsqueeze(-1).expand_as(diff)
                        l_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
                        internal_h_loss = internal_h_loss + l_weight * l_mse

                    # Normalize by number of layers to keep scale comparable to CE
                    internal_h_loss = internal_h_loss / (len(t_all_h) - 1)

                # Memory-efficient chunked head and loss distillation
                # Compare positions 0..S-2 predicting 1..S-1
                chunk_size = int(getattr(cfg, "loss_chunk_size", 32) or 32)
                kl, ce, t_agree = compute_chunked_logits_losses(
                    student=student,
                    teacher_provider=teacher_provider,
                    s_hidden=s_hidden,
                    t_hidden=t_hidden,
                    attention_mask=batch.attention_mask,
                    loss_mask=batch.loss_mask,
                    targets=batch.input_ids,
                    temperature=float(cur_temp),
                    kl_weight=(cur_kl_w if need_teacher_targets else 0.0),
                    ce_weight=cur_ce_w,
                    chunk_size=chunk_size,
                    device=device,
                )

                loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                feature_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                feature_breakdown: dict[str, float] = {}
                if getattr(cfg, "distill_enabled", False):
                    features = feature_extractor.extract(
                        wrappers=wrappers,
                        student_hidden_states=s_all_h,
                        teacher_hidden_states=t_all_h,
                    )
                    feature_loss, feature_breakdown = compute_feature_distill_loss(
                        features=features,
                        attention_mask=batch.attention_mask,
                        mse_weight=float(getattr(cfg, "distill_mse_weight", 0.0)),
                        cosine_weight=float(getattr(cfg, "distill_cosine_weight", 0.0)),
                        layer_weight_spec=getattr(cfg, "distill_layer_weights", None),
                        device=device,
                    )
                    loss = loss + feature_loss
                if cur_kl_w > 0.0:
                    loss = loss + (cur_kl_w * kl)

                if cur_ce_w > 0.0:
                    loss = loss + (cur_ce_w * ce)

                if cfg.hidden_match_weight > 0.0:
                    loss = loss + cfg.hidden_match_weight * internal_h_loss

                if getattr(cfg, "wda_gain_l2_weight", 0.0) > 0.0:
                    gain_target = float(getattr(cfg, "wda_gain_target", 1.0))
                    gain_l2 = sum(
                        (m.wda_gain - gain_target) ** 2 for m in wrappers
                    ) / len(patched_paths)
                    loss = loss + cfg.wda_gain_l2_weight * gain_l2

                # Routing entropy regularization
                if getattr(cfg, "routing_entropy_weight", 0.0) > 0.0:
                    ent_sum = 0.0
                    ent_count = 0
                    for m in wrappers:
                        ent = m.wda_attn.routing_stats.get("entropy", 0.0)
                        ent_sum += ent
                        ent_count += 1
                    if ent_count > 0:
                        loss = loss + cfg.routing_entropy_weight * (ent_sum / ent_count)

                # Track components for ultra-granular logging
                total_kl += float(kl.item())
                total_ce += float(ce.item())
                total_h += internal_h_loss.item()
                if getattr(cfg, "distill_enabled", False):
                    total_feature += float(feature_loss.item())
                    for k, v in feature_breakdown.items():
                        feature_breakdown_totals[k] = feature_breakdown_totals.get(
                            k, 0.0
                        ) + float(v)

                if cfg.gamma_l1_weight > 0.0:
                    gamma_l1 = sum(
                        torch.abs(m.gamma - cfg.gamma_l1_target) for m in wrappers
                    ) / len(patched_paths)
                    loss = loss + cfg.gamma_l1_weight * gamma_l1

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at step {step} (skipping step)")
                bad_step = True
                break

            total_loss += loss.item()
            loss_scaled = loss / float(cfg.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        if bad_step:
            opt.zero_grad(set_to_none=True)
            continue

        # Post-step bookkeeping
        avg_loss = total_loss / cfg.grad_accum
        avg_kl = total_kl / cfg.grad_accum
        avg_ce = total_ce / cfg.grad_accum
        avg_h = total_h / cfg.grad_accum
        avg_feature = total_feature / cfg.grad_accum if cfg.grad_accum > 0 else 0.0
        avg_feature_terms = {
            k: v / cfg.grad_accum for k, v in feature_breakdown_totals.items()
        }

        if autopilot:
            autopilot.update(avg_kl)

            # YELLOW ZONE: Pre-emptive stabilization
            # Trigger if KL is elevated OR agreement is dropping significantly
            if (
                avg_kl > 1.5 or (t_agree >= 0.0 and t_agree < 0.55)
            ) and autopilot.progress > 500:
                if yellow_zone_active == 0:
                    print(
                        f">>> YELLOW ALERT at step {step}: KL={avg_kl:.2f}, Agree={t_agree:.2f}. Softening knobs for 100 steps."
                    )
                yellow_zone_active = 100
                staged_kl_boost = 0.2
                staged_ce_reduction = 0.25
                staged_gt_softness = 0.05

            # RED ZONE: Hard Divergence detected
            # We use a 6.0 KL threshold as requested by the user, or 5x the target_high.
            hard_reject_threshold = max(6.0, autopilot.target_high * 5.0)
            if avg_kl > hard_reject_threshold and autopilot.progress > 100:
                reset_student_gammas(student, autopilot=autopilot)
                print(
                    f">>> SAFETY: Rejecting divergent batch at step {step} (KL={avg_kl:.2f})"
                )
                opt.zero_grad(set_to_none=True)
                continue

        # Prepare all trainable params for clipping
        if not static_trainable:
            all_trainable = [
                p for g in opt.param_groups for p in g["params"] if p.requires_grad
            ]

        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_trainable, cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(all_trainable, cfg.max_grad_norm)
            opt.step()

        # --- Clamping / Floor Enforcement (Post-Step) ---
        gamma_floor_start = float(getattr(cfg, "gamma_floor_start", 0.0))
        gamma_floor_end = float(getattr(cfg, "gamma_floor_end", gamma_floor_start))
        gamma_floor_start_step = int(getattr(cfg, "gamma_floor_start_step", 0))
        gamma_floor_steps = int(getattr(cfg, "gamma_floor_steps", 0))
        gamma_floor_topk = int(getattr(cfg, "gamma_floor_topk", 0) or 0)
        gamma_floor_focus_scale = float(getattr(cfg, "gamma_floor_focus_scale", 1.0))
        gamma_max = getattr(cfg, "gamma_max", None)
        if gamma_max is not None:
            gamma_max = float(gamma_max)
            if gamma_max <= 0.0:
                gamma_max = None
        wda_gain_min = getattr(cfg, "wda_gain_min", None)
        if wda_gain_min is not None:
            wda_gain_min = float(wda_gain_min)
        wda_gain_max = getattr(cfg, "wda_gain_max", None)
        if wda_gain_max is not None:
            wda_gain_max = float(wda_gain_max)

        p_g = _ramp(
            current_progress,
            start_step=gamma_floor_start_step,
            steps=gamma_floor_steps,
        )
        gamma_floor = gamma_floor_start + p_g * (gamma_floor_end - gamma_floor_start)
        gamma_floor = max(0.0, float(gamma_floor))

        topk_layers: set[int] = set()
        if gamma_floor_topk > 0:
            # We use a simple heuristic: if we are and many layers, just pick the early ones or do a quick scan
            # For simplicity, we'll use the cached stats if available from previous step or just all.
            # Realistically, we can just scan all wrappers' baseline_pct here.
            baseline_stats_pre = collect_baseline_stats(wrappers, include_rms=False)
            if baseline_stats_pre:
                sorted_pre = sorted(
                    baseline_stats_pre, key=lambda s: s["baseline_pct"], reverse=True
                )
                for s in sorted_pre[:gamma_floor_topk]:
                    if s["layer_idx"] is not None:
                        topk_layers.add(int(s["layer_idx"]))

        gamma_clamped = 0
        wda_gain_clamped = 0
        if (
            gamma_floor > 0.0
            or gamma_max is not None
            or wda_gain_min is not None
            or wda_gain_max is not None
        ):
            with torch.no_grad():
                for m in wrappers:
                    # Gamma Clamp
                    g_raw = float(m.gamma.item())
                    if gamma_floor > 0.0 or gamma_max is not None:
                        sign = -1.0 if g_raw < 0 else 1.0
                        target_min = gamma_floor
                        if m.layer_idx in topk_layers:
                            target_min = gamma_floor * gamma_floor_focus_scale
                        target_min = max(0.0, float(target_min))
                        g_abs = abs(g_raw)
                        new_abs = g_abs
                        if target_min > 0.0:
                            new_abs = max(new_abs, target_min)
                        if gamma_max is not None:
                            new_abs = min(new_abs, gamma_max)
                        new_val = sign * new_abs if new_abs != 0.0 else 0.0
                        if new_val != g_raw:
                            m.gamma.fill_(new_val)
                            gamma_clamped += 1

                    # Gain Clamp
                    g2_raw = float(m.wda_gain.item())
                    g2_val = g2_raw
                    if wda_gain_min is not None:
                        g2_val = max(g2_val, wda_gain_min)
                    if wda_gain_max is not None:
                        g2_val = min(g2_val, wda_gain_max)
                    if g2_val != g2_raw:
                        m.wda_gain.fill_(g2_val)
                        wda_gain_clamped += 1

        # EMA update (post-step)
        if ema_enabled and ema_params is not None:
            ema_decay = float(getattr(cfg, "ema_decay", 0.9995))
            one_minus = 1.0 - ema_decay
            with torch.no_grad():
                for name, p in student.named_parameters():
                    if not p.requires_grad:
                        continue
                    e = ema_params.get(name)
                    if e is None:
                        continue
                    e.mul_(ema_decay).add_(p.detach(), alpha=one_minus)

        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t0
            tok_s = tokens_seen / max(dt, 1e-6)
            mem = 0.0
            if device.startswith("cuda"):
                mem = torch.cuda.max_memory_allocated() / (1024**3)

            avg_ts = (
                sum(teacher_scales.values()) / len(teacher_scales)
                if teacher_scales
                else 0.0
            )

            baseline_stats = collect_baseline_stats(wrappers, include_rms=True)
            baseline_topk = int(getattr(cfg, "baseline_pct_log_topk", 5) or 5)
            (
                avg_baseline_pct,
                avg_abs_s,
                avg_eff_baseline_pct,
                avg_eff_s,
                avg_rms_ratio,
                avg_attn_rms,
                avg_wda_rms,
                avg_wda_gain,
                baseline_top,
            ) = summarize_baseline(baseline_stats, topk=baseline_topk)

            # Extra stats for debugging gamma movement
            gamma_vals = [s["gamma"] for s in baseline_stats]
            min_g = min(gamma_vals) if gamma_vals else 0.0
            max_g = max(gamma_vals) if gamma_vals else 0.0
            avg_g = sum(gamma_vals) / len(gamma_vals) if gamma_vals else 0.0

            eff_metrics = {}
            for s in baseline_stats:
                l_idx = s.get("layer_idx")
                if l_idx is None:
                    continue
                eff_metrics[f"layer_{l_idx}_attn_rms"] = s.get("attn_rms", 0.0)
                eff_metrics[f"layer_{l_idx}_wda_rms"] = s.get("wda_rms", 0.0)
                eff_metrics[f"layer_{l_idx}_rms_ratio"] = s.get("rms_ratio", 0.0)
                eff_metrics[f"layer_{l_idx}_eff_s"] = s.get("eff_s", 0.0)
                eff_metrics[f"layer_{l_idx}_eff_baseline_pct"] = s.get(
                    "eff_baseline_pct", 0.0
                )
                eff_metrics[f"layer_{l_idx}_wda_gain"] = s.get("wda_gain", 0.0)

            log_target = target_step if target_step is not None else cfg.steps
            drop_defaults = bool(getattr(cfg, "log_drop_defaults", False))
            drop_values = (
                cfg.log_drop_values if cfg.log_drop_values is not None else [0.0, 1.0]
            )
            drop_tol = float(getattr(cfg, "log_drop_tolerance", 1e-6))

            def _is_default(val) -> bool:
                if not drop_defaults:
                    return False
                if isinstance(val, bool):
                    return False
                if not isinstance(val, (int, float)):
                    return False
                v = float(val)
                for dv in drop_values:
                    try:
                        if abs(v - float(dv)) <= drop_tol:
                            return True
                    except Exception:
                        continue
                return False

            def _maybe_add(parts, label, val, fmt, *, weight=None, weight_fmt="{:.1f}"):
                if _is_default(val) and (weight is None or _is_default(weight)):
                    return
                if weight is None:
                    parts.append(f"{label}={fmt.format(val)}")
                else:
                    parts.append(
                        f"{label}={fmt.format(val)} (w={weight_fmt.format(weight)})"
                    )

            log_parts = [f"[{step:5d}/{log_target}]", f"loss={avg_loss:.4f}"]
            _maybe_add(log_parts, "kl", avg_kl, "{:.2f}", weight=cur_kl_w)
            _maybe_add(log_parts, "ce", avg_ce, "{:.2f}", weight=cur_ce_w)
            if t_agree >= 0:
                _maybe_add(log_parts, "agree", t_agree, "{:.2f}")
            _maybe_add(log_parts, "alpha", alpha, "{:.3f}")
            _maybe_add(log_parts, "ts", avg_ts, "{:.2f}")
            _maybe_add(log_parts, "base", avg_baseline_pct, "{:.2f}")
            _maybe_add(log_parts, "eff", avg_eff_baseline_pct, "{:.2f}")
            _maybe_add(log_parts, "|s|", avg_abs_s, "{:.4f}")
            _maybe_add(log_parts, "g_min", min_g, "{:.4f}")
            _maybe_add(log_parts, "g_max", max_g, "{:.4f}")
            _maybe_add(log_parts, "eff|s|", avg_eff_s, "{:.4f}")
            _maybe_add(log_parts, "rms", avg_rms_ratio, "{:.3f}")
            _maybe_add(log_parts, "gain", avg_wda_gain, "{:.3f}")
            _maybe_add(log_parts, "gt", current_gate_temp, "{:.3f}")
            if autopilot:
                log_parts.append(
                    f"prog={autopilot.progress:.1f} spd={autopilot.speed:.3f}"
                )
            if getattr(cfg, "distill_enabled", False):
                _maybe_add(log_parts, "distill", avg_feature, "{:.4f}")
            log_msg = " ".join(log_parts)
            print(log_msg)

            logger.log(
                step,
                {
                    "loss": avg_loss,
                    "kl": avg_kl,
                    "ce": avg_ce,
                    "h_loss": avg_h,
                    "distill_loss": avg_feature,
                    "alpha": alpha,
                    "avg_ts": avg_ts,
                    "avg_baseline_pct": avg_baseline_pct,
                    "avg_abs_s_scale": avg_abs_s,
                    "avg_eff_baseline_pct": avg_eff_baseline_pct,
                    "avg_eff_s_scale": avg_eff_s,
                    "avg_rms_ratio": avg_rms_ratio,
                    "avg_attn_rms": avg_attn_rms,
                    "avg_wda_rms": avg_wda_rms,
                    "avg_wda_gain": avg_wda_gain,
                    "baseline_top": baseline_top,
                    "gamma_floor": gamma_floor,
                    "gamma_max": gamma_max,
                    "gamma_clamped": gamma_clamped,
                    "wda_gain_min": wda_gain_min,
                    "wda_gain_max": wda_gain_max,
                    "wda_gain_clamped": wda_gain_clamped,
                    "gamma_floor_topk": gamma_floor_topk,
                    "gamma_floor_focus_scale": gamma_floor_focus_scale,
                    "gamma_min_val": min_g,
                    "gamma_max_val": max_g,
                    "gamma_avg_val": avg_g,
                    "tok_s": tok_s,
                    "vram_gb": mem,
                    # **eff_metrics,
                    # **avg_feature_terms,
                },
            )

            if (
                getattr(cfg, "eval_enabled", False)
                and int(getattr(cfg, "eval_every", 0) or 0) > 0
                and (step % int(cfg.eval_every) == 0)
            ):
                try:
                    run_eval(
                        model=student,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        device=device,
                        step=step,
                        out_dir=out_dir,
                    )
                except Exception as exc:
                    print(f"WARNING: eval harness failed at step {step}: {exc}")

        if step % cfg.save_every == 0:
            ckpt = out_dir / f"adapter_step{step}.pt"
            # Include autopilot state in payload
            payload_kwargs = {"step": step}
            if autopilot:
                payload_kwargs["autopilot_progress"] = autopilot.progress
                payload_kwargs["autopilot_speed"] = autopilot.speed

            save_adapter(student, ckpt, **payload_kwargs)

            # Optional: also save EMA snapshot once EMA mode is active
            if ema_enabled and ema_params is not None and use_ema_teacher:
                ema_ckpt = out_dir / f"adapter_step{step}_ema.pt"
                save_adapter_state(
                    ema_params,
                    ema_ckpt,
                    **payload_kwargs,
                )
            # Also save the config used.
            (out_dir / "config_used.json").write_text(
                Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"Saved {ckpt}")

        step += 1

    # Final save after loop completion
    # Optional: force absolute 1.0/0.0 for the final snapshot.
    final_force_teacher_off = bool(getattr(cfg, "final_force_teacher_off", True))

    if final_force_teacher_off:
        for m in wrappers:
            m.set_alpha(1.0)
            m.set_scales(teacher_scale=0.0, wda_scale=1.0)
            if hasattr(m.wda_attn, "gate_temp"):
                m.wda_attn.gate_temp = cfg.wda_gate_temp

    ckpt = out_dir / f"adapter_final.pt"
    payload_kwargs = {"step": step}
    if autopilot:
        payload_kwargs["autopilot_progress"] = autopilot.progress
        payload_kwargs["autopilot_speed"] = autopilot.speed
    save_adapter(student, ckpt, **payload_kwargs)
    print(f"Saved final checkpoint: {ckpt}")

    if ema_enabled and ema_params is not None:
        ema_ckpt = out_dir / "adapter_final_ema.pt"
        save_adapter_state(ema_params, ema_ckpt, **payload_kwargs)
        print(f"Saved EMA final checkpoint: {ema_ckpt}")
    logger.close()


if __name__ == "__main__":
    main()
