from __future__ import annotations

import torch
import torch.nn as nn


class ParallelAttentionWrapper(nn.Module):
    """Runs frozen teacher attention in parallel with trainable WDA.

    Output is teacher_attn + alpha * gamma * wda_delta.

    This keeps HF cache behavior intact via the teacher branch.
    """

    collect_stats: bool = False
    collect_features: bool = False
    feature_detach: bool = True

    def __init__(
        self,
        teacher_attn: nn.Module,
        wda_attn: nn.Module,
        *,
        layer_idx: int | None = None,
        init_alpha: float = 0.0,
        init_teacher_scale: float = 1.0,
        init_wda_scale: float = 1.0,
        init_layer_alpha_scale: float = 1.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        self.teacher_attn = teacher_attn
        self.wda_attn = wda_attn
        self.layer_idx = layer_idx
        self.gamma = nn.Parameter(torch.full((), float(init_gamma)))
        self.wda_gain = nn.Parameter(torch.full((), 1.0))
        self.wda_cache_enabled = False
        self.register_buffer("alpha", torch.tensor(float(init_alpha)), persistent=False)
        self.register_buffer(
            "teacher_scale", torch.tensor(float(init_teacher_scale)), persistent=False
        )
        self.register_buffer(
            "wda_scale", torch.tensor(float(init_wda_scale)), persistent=False
        )
        self.register_buffer(
            "layer_alpha_scale",
            torch.tensor(float(init_layer_alpha_scale)),
            persistent=False,
        )
        self.last_attn_rms = None
        self.last_wda_rms = None
        self.last_attn_out = None
        self.last_wda_out = None
        self.last_mixed_out = None

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

    def set_scales(
        self, *, teacher_scale: float | None = None, wda_scale: float | None = None
    ) -> None:
        if teacher_scale is not None:
            self.teacher_scale[...] = float(teacher_scale)
        if wda_scale is not None:
            self.wda_scale[...] = float(wda_scale)

    @property
    def routing_stats(self) -> dict[str, float]:
        if hasattr(self.wda_attn, "routing_stats"):
            return self.wda_attn.routing_stats
        return {}

    def _ensure_wda_on_device(self, device, dtype):
        # Only call next(parameters) once to avoid repeated syncs
        try:
            p = next(self.wda_attn.parameters())
            if p.device != device or p.dtype != dtype:
                self.wda_attn.to(device=device, dtype=dtype)
        except StopIteration:
            pass

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # Move scales to match hidden_states device/dtype once
        # Using .detach() to ensure no gradient through scales if not intended
        ts = self.teacher_scale.to(hidden_states.device, hidden_states.dtype)
        ws = self.wda_scale.to(hidden_states.device, hidden_states.dtype)
        a = (self.alpha * self.layer_alpha_scale).to(
            hidden_states.device, hidden_states.dtype
        )
        g = self.gamma.to(hidden_states.device, hidden_states.dtype)
        w = self.wda_gain.to(hidden_states.device, hidden_states.dtype)

        # Optimization: use float() on the buffer directly to avoid GPU sync if it was updated on CPU
        # But since they are buffers, we can just check them.
        # To be safe and fast, we check the underlying tensor value.
        ts_val = self.teacher_scale.item()
        ws_val = self.wda_scale.item()
        a_val = self.alpha.item()

        # 1. Compute Teacher only if scale > 0 or if we need KV cache for generation
        teacher_out = None
        needs_cache = kwargs.get("use_cache", False) or (
            kwargs.get("past_key_value") is not None
        )

        # IMPORTANT: If teacher_scale is effectively zero, KV caching via the
        # teacher branch is unavailable. WDA cache can be enabled separately.
        if needs_cache and ts_val <= 1e-20 and not self.wda_cache_enabled:
            raise RuntimeError(
                "Teacher attention is disabled (teacher_scale≈0) but KV caching was requested. "
                "Enable WDA cache or set use_cache=False for WDA-only inference."
            )

        # We run teacher if ts > 0 OR if we need to update the cache.
        # Using a microscopic threshold for the 'ghost anchor' to prevent logical discontinuities.
        if ts_val > 1e-20 or (needs_cache and not self.wda_cache_enabled):
            teacher_out = self.teacher_attn(hidden_states, *args, **kwargs)

        # 2. Extract internal states if teacher was run
        if teacher_out is not None:
            if isinstance(teacher_out, tuple):
                attn_out = teacher_out[0]
                rest = teacher_out[1:]
            else:
                attn_out = teacher_out
                rest = None
        else:
            # If teacher skipped (no ts and no cache needed),
            # we return 0 for attention and dummy weights.
            attn_out = torch.zeros_like(hidden_states)
            # Most models expect at least (output, weights)
            rest = (None,)

        # 3. Compute WDA branch
        # If WDA is effectively disabled, avoid computing it
        wda_out = None
        if ws_val < 1e-9 or a_val < 1e-9:
            mixed = ts * attn_out
        else:
            # Optimized device/dtype check
            self._ensure_wda_on_device(hidden_states.device, hidden_states.dtype)

            wda_out = self.wda_attn(hidden_states, *args, **kwargs)

            # Variance-Preserving Fusion:
            # We treat the Teacher and Student as vectors and normalize the sum.
            # This prevents the 'volume expansion' that causes KL divergence at 30 layers.
            # denom = sqrt(teacher_scale^2 + (wda_scale * alpha * gamma)^2)
            s_scale = ws * a * g * w
            denom = torch.sqrt(ts**2 + s_scale**2 + 1e-12)
            mixed = (ts * attn_out + s_scale * wda_out) / denom

        if ParallelAttentionWrapper.collect_stats:
            with torch.no_grad():
                attn_rms = 0.0
                wda_rms = 0.0
                try:
                    attn_rms = torch.sqrt(attn_out.float().pow(2).mean()).item()
                except Exception:
                    attn_rms = 0.0
                if wda_out is not None:
                    try:
                        wda_rms = torch.sqrt(wda_out.float().pow(2).mean()).item()
                    except Exception:
                        wda_rms = 0.0
                self.last_attn_rms = attn_rms
                self.last_wda_rms = wda_rms

        if ParallelAttentionWrapper.collect_features:
            attn_feat = attn_out
            wda_feat = wda_out
            mixed_feat = mixed
            if ParallelAttentionWrapper.feature_detach:
                attn_feat = attn_feat.detach()
                if wda_feat is not None:
                    wda_feat = wda_feat.detach()
                mixed_feat = mixed_feat.detach()
            self.last_attn_out = attn_feat
            self.last_wda_out = wda_feat
            self.last_mixed_out = mixed_feat

        if rest is None:
            return mixed

        # If teacher was skipped and WDA cache is enabled, emit a cache tuple
        # in the position expected by HF generation.
        if teacher_out is None and needs_cache and self.wda_cache_enabled:
            output_attentions = bool(kwargs.get("output_attentions", False))
            wda_cache = getattr(self.wda_attn, "last_cache", None)
            if output_attentions:
                return (mixed, None, wda_cache)
            return (mixed, wda_cache)

        return (mixed,) + rest
