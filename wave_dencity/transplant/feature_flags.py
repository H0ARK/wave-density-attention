from __future__ import annotations

from wave_dencity.transplant.config import TransplantConfig


def apply_feature_flags(cfg: TransplantConfig) -> TransplantConfig:
    run_mode = getattr(cfg, "run_mode", None)
    teacher_type = getattr(cfg, "teacher_type", None)
    student_attn_impl = getattr(cfg, "student_attn_impl", None)

    if run_mode == "teacher_free":
        cfg.teacher_free_mode = True
    elif run_mode in {"mixing_bridge", "teacher_target_distill"}:
        cfg.teacher_free_mode = False

    if teacher_type == "none":
        if run_mode == "teacher_target_distill":
            raise ValueError(
                "teacher_type='none' is incompatible with run_mode='teacher_target_distill'"
            )
        cfg.teacher_free_mode = True
        cfg.ema_teacher_enabled = False
    elif teacher_type == "ema_student":
        cfg.teacher_free_mode = False
        cfg.ema_teacher_enabled = True

    if student_attn_impl == "wda":
        cfg.alpha_max = 1.0
        if cfg.teacher_free_mode:
            cfg.teacher_free_teacher_scale = 0.0
            cfg.teacher_free_wda_scale = 1.0
        else:
            cfg.teacher_scale_start = 0.0
            cfg.teacher_scale_end = 0.0
            cfg.wda_scale_start = 1.0
            cfg.wda_scale_end = 1.0
    elif student_attn_impl == "baseline":
        cfg.alpha_max = 0.0
        if cfg.teacher_free_mode:
            cfg.teacher_free_teacher_scale = 1.0
            cfg.teacher_free_wda_scale = 0.0
        else:
            cfg.teacher_scale_start = 1.0
            cfg.teacher_scale_end = 1.0
            cfg.wda_scale_start = 0.0
            cfg.wda_scale_end = 0.0

    return cfg
