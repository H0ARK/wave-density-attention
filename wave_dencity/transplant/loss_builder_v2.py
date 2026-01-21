from __future__ import annotations

from typing import Any
import re

import torch
import torch.nn.functional as F


def _layer_weights(num_layers: int, spec: str | list[float] | None) -> list[float]:
    if num_layers <= 0:
        return []
    if spec is None or spec == "uniform":
        return [1.0 for _ in range(num_layers)]
    if spec == "depth_decay":
        return [1.0 / (i + 1) for i in range(num_layers)]
    if isinstance(spec, list) and len(spec) == num_layers:
        return [float(v) for v in spec]
    return [1.0 for _ in range(num_layers)]


_LAYER_KEY_RE = re.compile(r"layer_(\d+)")


def _layer_idx_from_key(key: str) -> int | None:
    m = _LAYER_KEY_RE.search(key)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (a - b).pow(2)
    if diff.dim() == 2:
        return (diff * mask).sum() / (mask.sum() + 1e-8)
    # Average over hidden dimension as well if present (typical for [B, S, D])
    mask = mask.unsqueeze(-1).expand_as(diff)
    total_elements = mask.sum() + 1e-8
    return (diff * mask).sum() / total_elements


def _masked_cosine(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if a.dim() == 2:
        # For [B, S] heatmaps, cosine is just on scalars which is not very useful
        # but we treat the whole sequence as a vector.
        a_flat = a * mask
        b_flat = b * mask
        # Vectorize across all valid tokens
        return (
            1.0
            - F.cosine_similarity(a_flat.view(1, -1), b_flat.view(1, -1), dim=-1).mean()
        )

    # For [B, S, D] hidden states, compute cosine per token and average.
    a_flat = a.reshape(-1, a.size(-1))
    b_flat = b.reshape(-1, b.size(-1))
    mask_flat = mask.reshape(-1)
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return 1.0 - (cos * mask_flat).sum() / (mask_flat.sum() + 1e-8)


def compute_feature_distill_loss(
    *,
    features: dict[str, dict[str, Any]],
    attention_mask: torch.Tensor,
    mse_weight: float,
    cosine_weight: float,
    layer_weight_spec: str | list[float] | None,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    student = features.get("student", {})
    teacher = features.get("teacher", {})

    losses: dict[str, float] = {}
    total = torch.tensor(0.0, device=device, dtype=torch.float32)

    if "hidden_states" in student and "hidden_states" in teacher:
        s_all = student["hidden_states"]
        t_all = teacher["hidden_states"]
        n_layers = min(len(s_all), len(t_all)) - 1
        weights = _layer_weights(n_layers, layer_weight_spec)
        w_sum = float(sum(weights)) if weights else 1.0
        h_mse = torch.tensor(0.0, device=device, dtype=torch.float32)
        h_cos = torch.tensor(0.0, device=device, dtype=torch.float32)
        for i in range(1, n_layers + 1):
            s_h = s_all[i]
            t_h = t_all[i]
            w = weights[i - 1]
            if mse_weight > 0.0:
                h_mse = h_mse + w * _masked_mse(s_h, t_h, attention_mask)
            if cosine_weight > 0.0:
                h_cos = h_cos + w * _masked_cosine(s_h, t_h, attention_mask)
        if mse_weight > 0.0:
            if w_sum > 0.0:
                h_mse = h_mse / w_sum
            total = total + mse_weight * h_mse
            losses["hidden_mse"] = float(h_mse.item())
        if cosine_weight > 0.0:
            if w_sum > 0.0:
                h_cos = h_cos / w_sum
            total = total + cosine_weight * h_cos
            losses["hidden_cosine"] = float(h_cos.item())

    layer_items: list[tuple[str, torch.Tensor, torch.Tensor, int | None]] = []
    max_idx = -1
    for key, s_feat in student.items():
        if not key.startswith("layer_"):
            continue
        if key not in teacher:
            continue
        t_feat = teacher[key]
        if not isinstance(s_feat, torch.Tensor) or not isinstance(t_feat, torch.Tensor):
            continue
        idx = _layer_idx_from_key(key)
        if idx is not None:
            max_idx = max(max_idx, idx)
        layer_items.append((key, s_feat, t_feat, idx))

    layer_weights = (
        _layer_weights(max_idx + 1, layer_weight_spec) if max_idx >= 0 else None
    )
    layer_mse = torch.tensor(0.0, device=device, dtype=torch.float32)
    layer_cos = torch.tensor(0.0, device=device, dtype=torch.float32)
    layer_mse_wsum = 0.0
    layer_cos_wsum = 0.0

    for key, s_feat, t_feat, idx in layer_items:
        w = 1.0
        if layer_weights is not None and idx is not None and idx < len(layer_weights):
            w = float(layer_weights[idx])
        if mse_weight > 0.0:
            mse_val = _masked_mse(s_feat, t_feat, attention_mask)
            layer_mse = layer_mse + w * mse_val
            layer_mse_wsum += w
            losses[f"{key}_mse"] = float(mse_val.item())
        if cosine_weight > 0.0:
            cos_val = _masked_cosine(s_feat, t_feat, attention_mask)
            layer_cos = layer_cos + w * cos_val
            layer_cos_wsum += w
            losses[f"{key}_cos"] = float(cos_val.item())

    if mse_weight > 0.0 and layer_mse_wsum > 0.0:
        total = total + mse_weight * (layer_mse / float(layer_mse_wsum))
    if cosine_weight > 0.0 and layer_cos_wsum > 0.0:
        total = total + cosine_weight * (layer_cos / float(layer_cos_wsum))

    return total, losses
