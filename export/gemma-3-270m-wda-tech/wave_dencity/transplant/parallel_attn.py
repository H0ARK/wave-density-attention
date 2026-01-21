from __future__ import annotations

import torch
import torch.nn as nn


class ParallelAttentionWrapper(nn.Module):
    """Runs frozen teacher attention in parallel with trainable WDA.

    Output is teacher_attn + alpha * gamma * wda_delta.

    This keeps HF cache behavior intact via the teacher branch.
    """

    def __init__(
        self,
        teacher_attn: nn.Module,
        wda_attn: nn.Module,
        *,
        init_alpha: float = 0.0,
        init_teacher_scale: float = 1.0,
        init_wda_scale: float = 1.0,
        init_layer_alpha_scale: float = 1.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        self.teacher_attn = teacher_attn
        self.wda_attn = wda_attn
        self.gamma = nn.Parameter(torch.full((), float(init_gamma)))
        self.register_buffer("alpha", torch.tensor(float(init_alpha)), persistent=False)
        self.register_buffer("teacher_scale", torch.tensor(float(init_teacher_scale)), persistent=False)
        self.register_buffer("wda_scale", torch.tensor(float(init_wda_scale)), persistent=False)
        self.register_buffer("layer_alpha_scale", torch.tensor(float(init_layer_alpha_scale)), persistent=False)

    def __getattr__(self, name: str):
        # Some HF decoder blocks read attributes off the attention module
        # (e.g. Gemma3 uses `self.self_attn.is_sliding`). Delegate missing
        # attributes to the wrapped teacher attention.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.teacher_attn, name)

    def set_alpha(self, value: float) -> None:
        self.alpha[...] = float(value)

    def set_scales(self, *, teacher_scale: float | None = None, wda_scale: float | None = None) -> None:
        if teacher_scale is not None:
            self.teacher_scale[...] = float(teacher_scale)
        if wda_scale is not None:
            self.wda_scale[...] = float(wda_scale)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        with torch.no_grad():
            teacher_out = self.teacher_attn(hidden_states, *args, **kwargs)

        if isinstance(teacher_out, tuple):
            attn_out = teacher_out[0]
            rest = teacher_out[1:]
        else:
            attn_out = teacher_out
            rest = None

        ts = self.teacher_scale.to(attn_out.dtype)
        ws = self.wda_scale.to(attn_out.dtype)
        a = (self.alpha * self.layer_alpha_scale).to(attn_out.dtype)
        g = self.gamma.to(attn_out.dtype)

        # If WDA is effectively disabled, avoid computing it (important for sanity checks
        # and also avoids device-mismatch issues if WDA hasn't been moved yet).
        if float(ws.detach().cpu().item()) == 0.0 or float(a.detach().cpu().item()) == 0.0:
            mixed = ts * attn_out
        else:
            # Lazy device/dtype move for WDA branch (helps MPS runs).
            try:
                wda_param = next(self.wda_attn.parameters())
                needs_move = (wda_param.device != hidden_states.device) or (wda_param.dtype != hidden_states.dtype)
            except StopIteration:
                needs_move = False

            if needs_move:
                self.wda_attn.to(device=hidden_states.device, dtype=hidden_states.dtype)

            wda_out = self.wda_attn(hidden_states, *args, **kwargs)
            mixed = ts * attn_out + (ws * a * g) * wda_out

        if rest is None:
            return mixed
        return (mixed,) + rest
