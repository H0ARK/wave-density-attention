from __future__ import annotations

import torch
import torch.nn as nn

from wave_dencity.model import WaveDensityAttentionBlock


class WDABridge(nn.Module):
    """Wraps the repo's WaveDensityAttentionBlock into an HF-attention-like delta.

    Returns a tensor shaped like an attention output: [B, S, D].
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        seq_len: int,
        num_masks: int,
        num_waves_per_mask: int,
        topk_masks: int,
        gate_temp: float = 1.0,
        attn_alpha: float,
        content_rank: int = 8,
        content_mix: float = 0.0,
        learned_content: bool = False,
        use_sin_waves: bool = True,
        use_sampling: bool = False,
        num_samples: int = 64,
        noise_sigma: float = 0.12,
        step_alpha: float = 6.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.block = WaveDensityAttentionBlock(
            embed_dim=int(hidden_size),
            seq_len=int(seq_len),
            num_heads=int(num_heads),
            num_masks=int(num_masks),
            num_waves_per_mask=int(num_waves_per_mask),
            topk_masks=int(topk_masks),
            gate_temp=float(gate_temp),
            attn_alpha=float(attn_alpha),
            content_rank=int(content_rank),
            content_mix=float(content_mix),
            learned_content=bool(learned_content),
            use_sin_waves=bool(use_sin_waves),
            use_sampling=bool(use_sampling),
            num_samples=int(num_samples),
            noise_sigma=float(noise_sigma),
            step_alpha=float(step_alpha),
            use_checkpoint=bool(use_checkpoint),
            use_ffn=False,
        )
        self.last_cache = None

    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        # The underlying block returns (attn_out + x) when use_ffn=False.
        # Convert to an attention-like delta output.
        y = self.block(hidden_states, **_)
        if isinstance(y, tuple):
            out, cache = y
            self.last_cache = cache
            return out - hidden_states
        self.last_cache = None
        return y - hidden_states

    @property
    def routing_stats(self) -> dict[str, float]:
        return self.block.routing_stats
