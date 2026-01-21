from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only trainable transplant parameters (WDA + gamma).

    This keeps checkpoints small vs saving full HF weights.
    """
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        state[name] = param.detach().cpu()
    return state


def save_adapter(
    model: nn.Module, out_path: str | Path, *, step: int | None = None, **kwargs
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "adapter": adapter_state_dict(model),
        "payload": kwargs,
    }
    torch.save(payload, out_path)


def save_adapter_state(
    adapter_state: dict[str, torch.Tensor],
    out_path: str | Path,
    *,
    step: int | None = None,
    **kwargs,
) -> None:
    """Save an adapter checkpoint from a provided state dict.

    This is useful for exporting EMA (mean-teacher) weights without having to
    temporarily overwrite live model parameters.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "adapter": {k: v.detach().cpu() for k, v in adapter_state.items()},
        "payload": kwargs,
    }
    torch.save(payload, out_path)


def load_adapter(model: nn.Module, path: str | Path, *, strict: bool = False) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    adapter = (
        payload["adapter"]
        if isinstance(payload, dict) and "adapter" in payload
        else payload
    )
    model_state = model.state_dict()

    print(
        f"DEBUG load_adapter: path={path} adapter_keys={len(adapter)} model_keys={len(model_state)}"
    )

    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    shape_mismatch: list[str] = []
    for k, v in adapter.items():
        if k not in model_state:
            skipped.append(k)
            continue
        if tuple(model_state[k].shape) != tuple(v.shape):
            shape_mismatch.append(k)
            continue
        filtered[k] = v

    print(
        f"DEBUG load_adapter: filtered={len(filtered)} skipped={len(skipped)} shape_mismatch={len(shape_mismatch)}"
    )
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return {
        "loaded": len(filtered),
        "skipped": len(skipped),
        "shape_mismatch": len(shape_mismatch),
        "missing": missing,
        "unexpected": unexpected,
        "payload": payload,
    }


def set_parallel_attn_mode(
    model: nn.Module,
    *,
    alpha: float,
    teacher_scale: float,
    wda_scale: float = 1.0,
    gate_temp: float | None = None,
) -> None:
    """Set runtime mixing scales for all ParallelAttentionWrapper modules.

    Notes:
    - Setting teacher_scale=0.0 disables the teacher output contribution.
    - If you intend to run cacheless inference and reclaim VRAM, combine
      teacher_scale=0.0 with `offload_teacher_attention(..., device="cpu")`
      and ensure `use_cache=False` at call/generation time.
    """

    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.set_alpha(float(alpha))
            m.set_scales(teacher_scale=float(teacher_scale), wda_scale=float(wda_scale))
            if gate_temp is not None and hasattr(m.wda_attn, "gate_temp"):
                m.wda_attn.gate_temp = float(gate_temp)


def offload_teacher_attention(model: nn.Module, *, device: str = "cpu") -> int:
    """Move the *teacher attention branch* inside ParallelAttentionWrapper to `device`.

    This reduces GPU VRAM usage once teacher_scale==0 (or tiny) and you are not
    using KV-cache via the teacher branch.

    Returns:
        The number of wrappers updated.
    """

    n = 0
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.teacher_attn.to(device)
            n += 1
    return n
