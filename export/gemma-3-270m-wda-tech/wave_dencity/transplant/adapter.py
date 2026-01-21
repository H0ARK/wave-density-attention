from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


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


def save_adapter(model: nn.Module, out_path: str | Path, *, step: int | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "adapter": adapter_state_dict(model),
    }
    torch.save(payload, out_path)


def load_adapter(model: nn.Module, path: str | Path, *, strict: bool = False) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    adapter = payload["adapter"] if isinstance(payload, dict) and "adapter" in payload else payload
    model_state = model.state_dict()

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

    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return {
        "loaded": len(filtered),
        "skipped": len(skipped),
        "shape_mismatch": len(shape_mismatch),
        "missing": missing,
        "unexpected": unexpected,
        "payload": payload,
    }
