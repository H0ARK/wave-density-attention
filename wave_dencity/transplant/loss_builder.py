from __future__ import annotations

import torch
import torch.nn.functional as F

from wave_dencity.transplant.teacher_provider import TeacherProvider


def compute_chunked_logits_losses(
    *,
    student: torch.nn.Module,
    teacher_provider: TeacherProvider,
    s_hidden: torch.Tensor,
    t_hidden: torch.Tensor | None,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    targets: torch.Tensor,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    t_view = (
        t_hidden[:, :-1, :].reshape(-1, t_hidden.size(-1))
        if t_hidden is not None
        else None
    )
    s_view = s_hidden[:, :-1, :].reshape(-1, s_hidden.size(-1))
    mask_view = attention_mask[:, 1:].reshape(-1)

    if loss_mask is not None:
        l_mask = loss_mask[:, 1:].reshape(-1)
        mask_view = mask_view * l_mask

    targets_view = targets[:, 1:].reshape(-1)

    kl_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    ce_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    n_toks = 0
    t_agree = -1.0
    last_s_logits = None
    last_t_logits = None

    for i in range(0, s_view.size(0), chunk_size):
        c_mask = mask_view[i : i + chunk_size]
        if not c_mask.any():
            continue

        s_h_chunk = s_view[i : i + chunk_size][c_mask > 0]
        t_h_chunk = (
            t_view[i : i + chunk_size][c_mask > 0] if t_view is not None else None
        )
        target_chunk = targets_view[i : i + chunk_size][c_mask > 0]

        s_logits_chunk = student.lm_head(s_h_chunk).float()
        t_logits_chunk = None
        if kl_weight > 0.0 and t_h_chunk is not None:
            t_logits_chunk = teacher_provider.logits_from_hidden(t_h_chunk)
            if t_logits_chunk is not None:
                s_log_probs = F.log_softmax(s_logits_chunk / temperature, dim=-1)
                t_log_probs = F.log_softmax(t_logits_chunk / temperature, dim=-1)
                kl_chunk = F.kl_div(
                    s_log_probs,
                    t_log_probs,
                    reduction="sum",
                    log_target=True,
                )
                kl_sum = kl_sum + kl_chunk

        if ce_weight > 0.0:
            ce_chunk = F.cross_entropy(s_logits_chunk, target_chunk, reduction="sum")
            ce_sum = ce_sum + ce_chunk

        n_toks = n_toks + s_h_chunk.size(0)
        if t_logits_chunk is not None:
            last_s_logits = s_logits_chunk
            last_t_logits = t_logits_chunk

    if last_t_logits is not None:
        s_top = last_s_logits.argmax(dim=-1)
        t_top = last_t_logits.argmax(dim=-1)
        t_agree = (s_top == t_top).float().mean().item()

    if n_toks > 0:
        kl = kl_sum * (temperature * temperature) / (n_toks + 1e-8)
        ce = ce_sum / (n_toks + 1e-8)
    else:
        kl = torch.tensor(0.0, device=device, dtype=torch.float32)
        ce = torch.tensor(0.0, device=device, dtype=torch.float32)

    return kl, ce, t_agree
