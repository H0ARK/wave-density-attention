"""Wave-Density Attention (WDA) reference implementation.

This file is intended to be the *shippable* version of the model code:
- Minimal, readable, and self-contained.
- Uses streaming datasets (Hugging Face) so large corpora are not stored in-repo.
- Keeps checkpoint compatibility with the rest of this repo.

**Latent Thought Units (Chunk-level Routing):**
The model now supports internal chunk-level "thought units" that group tokens into
coherent segments (roughly sentence-level, ≤64 tokens by default). All tokens within
a chunk share the same wave-mask routing in attention, improving coherence and efficiency.

Key features:
- Chunks are computed internally during forward pass (latent, not visible in output)
- Routing (gating + modulation) is computed once per chunk and broadcast to all tokens
- Optional chunk consistency regularization encourages similar embeddings within chunks
- No changes to output format or visible reasoning traces

Primary entrypoint:
  python3 wave_dencity/model_thought.py train --dataset ultrachat --steps 10000

Notes:
- The core mechanism is WaveDensityAttentionBlock, which generates a *Toeplitz*
  (relative-position) causal attention kernel from wave interference and applies
  it via FFT-based causal convolution.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.compile / Dynamo can be finicky around torch.fft on some setups.
# We keep FFT helpers outside compile using dynamo_disable when available.
try:
    from torch._dynamo import disable as dynamo_disable  # type: ignore
except Exception:  # pragma: no cover

    def dynamo_disable(fn):
        return fn


class StraightThroughHeaviside(nn.Module):
    """Binary step function with a smooth surrogate gradient.

    Forward uses a hard threshold; backward uses sigmoid(alpha*x).
    """

    def __init__(self, alpha: float = 6.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hard = (x > 0).to(x.dtype)
        y_soft = torch.sigmoid(self.alpha * x)
        return y_hard + (y_soft - y_hard).detach()


def triangle_wave(x: torch.Tensor) -> torch.Tensor:
    """Triangle wave in [-1, 1] with period 2π."""
    t = (x - (math.pi / 2)) / (2 * math.pi)
    frac = t - torch.floor(t)
    tri01 = 1.0 - 2.0 * torch.abs(frac - 0.5)
    return 2.0 * tri01 - 1.0


def topk_sparse_weights(
    logits: torch.Tensor, k: int, temperature: float = 1.0
) -> torch.Tensor:
    """Sparse mask weights using a straight-through top-k selection.

    Args:
        logits: [B, M]
        k: number of nonzeros per row
        temperature: softmax temperature for gradient path

    Returns:
        weights: [B, M] with exactly top-k nonzeros per row, summing to 1.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    soft = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)

    topk_idx = torch.topk(soft, k=min(int(k), soft.shape[-1]), dim=-1).indices
    hard = torch.zeros_like(soft)
    hard.scatter_(dim=-1, index=topk_idx, value=1.0)
    hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-8)

    return hard + (soft - hard).detach()


def segment_into_chunks(
    seq_len: int,
    chunk_size: int = 64,
    tokens: torch.Tensor | None = None,
    tokenizer=None,
    use_punctuation: bool = False,
) -> torch.Tensor:
    """Segment a sequence into latent thought unit chunks.

    Chunks represent internal coherent units (roughly sentence-level) used to
    share attention routing and improve training stability. These are NOT
    visible reasoning traces or special tokens.

    Args:
        seq_len: sequence length
        chunk_size: max tokens per chunk (default: 64)
        tokens: [B, S] token IDs (optional, for punctuation-based chunking)
        tokenizer: tokenizer instance (optional, for punctuation detection)
        use_punctuation: if True and tokens/tokenizer provided, split on sentence boundaries

    Returns:
        chunk_ids: [B, S] integer chunk IDs, or [S] if tokens is None
    """
    if tokens is None or not use_punctuation:
        # Fixed-size chunking: simple and deterministic
        if tokens is not None:
            device = tokens.device
            chunk_ids = (
                (torch.arange(seq_len, device=device) // chunk_size)
                .unsqueeze(0)
                .expand(tokens.shape[0], -1)
            )
        else:
            # No tokens provided: return a 1D chunk id tensor on CPU
            chunk_ids = torch.arange(seq_len, device="cpu") // chunk_size
        return chunk_ids

    # Punctuation-based chunking (Vectorized)
    B, S = tokens.shape
    if tokenizer is None:
        return (
            (torch.arange(S, device=tokens.device) // chunk_size)
            .unsqueeze(0)
            .expand(B, -1)
        )

    # Detect sentence boundaries
    period_ids = []
    for punct in [".", "!", "?", "\n"]:
        enc = tokenizer.encode(punct, add_special_tokens=False)
        period_ids.extend(enc)
    period_ids_t = torch.tensor(list(set(period_ids)), device=tokens.device)

    # is_punct: [B, S]
    is_punct = torch.isin(tokens, period_ids_t)

    # We want to split if is_punct OR we hit chunk_size.
    # To avoid many tiny chunks, we can use a simpler vectorized heuristic:
    # 1. Base chunks every chunk_size
    # 2. Add an offset every time we hit a punctuation

    # This creates a unique chunk ID that increments on either event.
    # Note: this might create chunks larger than chunk_size if punctuation is sparse,
    # but the base_chunks fixed split ensures a maximum size of chunk_size *anyway*.
    base_chunks = torch.arange(S, device=tokens.device) // chunk_size
    punct_offsets = torch.cumsum(is_punct.long(), dim=1)

    # Shift punct_offsets by 1 to make the boundary inclusive of the punct
    punct_offsets = torch.cat(
        [
            torch.zeros((B, 1), device=tokens.device, dtype=torch.long),
            punct_offsets[:, :-1],
        ],
        dim=1,
    )

    chunk_ids = base_chunks.unsqueeze(0) + punct_offsets
    return chunk_ids


class WaveDensityAttentionBlock(nn.Module):
    """Causal self-attention using wave interference + density formation.

    Instead of QK^T logits, we generate a *relative-position* logit kernel via a
    gated mixture of wave masks. Because the kernel depends only on offset d=i-j,
    attention is Toeplitz and can be applied with causal convolution.

    The block supports a small auxiliary content-mixing branch (optional) to
    improve topicality while keeping cost linear.

    Chunk-level routing (thought units):
    When chunk_ids are provided, the gating network pools embeddings per chunk
    rather than per sequence. All tokens in a chunk share the same wave-mask
    routing, improving coherence and reducing compute.
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        *,
        num_heads: int = 4,
        num_masks: int = 16,
        num_waves_per_mask: int = 8,
        topk_masks: int = 8,
        gate_hidden: int = 128,
        attn_alpha: float = 3.0,
        content_rank: int = 8,
        content_mix: float = 0.15,
        learned_content: bool = True,
        use_sin_waves: bool = True,
        use_sampling: bool = False,
        num_samples: int = 64,
        noise_sigma: float = 0.12,
        step_alpha: float = 6.0,
        use_checkpoint: bool = False,
        use_ffn: bool = True,
        ffn_mult: int = 4,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_masks = int(num_masks)
        self.num_waves_per_mask = int(num_waves_per_mask)
        self.topk_masks = int(topk_masks)

        self.attn_alpha = float(attn_alpha)
        self.use_sampling = bool(use_sampling)
        self.num_samples = int(num_samples)
        self.noise_sigma = float(noise_sigma)
        self.step = StraightThroughHeaviside(alpha=step_alpha)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_sin_waves = bool(use_sin_waves)

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Lightweight per-token modulation of values.
        self.q_mod = nn.Linear(self.embed_dim, self.num_heads)
        self.k_mod = nn.Linear(self.embed_dim, self.num_heads)

        # Per-head gating selects which wave masks are active.
        self.gatings = nn.Linear(self.embed_dim, self.num_heads * self.num_masks)

        # Per-head, low-rank content-conditioned modulation of mask weights.
        self.mod_rank = 8
        self.mod_mlp = nn.Linear(self.embed_dim, self.num_heads * self.mod_rank)
        self.mod_basis = nn.Parameter(
            torch.randn(self.mod_rank, self.num_heads, self.num_masks) * 0.05
        )

        # Wave parameters for 1D relative-position kernel.
        self.freqs = nn.Parameter(
            torch.randn(self.num_heads, self.num_masks, self.num_waves_per_mask, 2)
            * 1.5
        )
        self.amps = nn.Parameter(
            torch.randn(self.num_heads, self.num_masks, self.num_waves_per_mask) * 0.02
            + 0.5
        )
        self.phases = nn.Parameter(
            torch.randn(self.num_heads, self.num_masks, self.num_waves_per_mask)
            * math.pi
        )

        # Relative position encoding for lags in [-(S-1), ..., (S-1)] => length 2S-1
        rel = torch.arange(-(self.seq_len - 1), self.seq_len).float()
        p1 = rel / self.seq_len
        p2 = (
            torch.sign(rel)
            * torch.sqrt(torch.abs(rel) + 1e-6)
            / math.sqrt(self.seq_len)
        )
        pos_kern = torch.stack([p1, p2], dim=-1)  # [2S-1,2]
        self.register_buffer("pos_kern", pos_kern)

        # Windowing to bias toward local interactions without hard cutoffs.
        dpos = torch.arange(0, self.seq_len).float()
        window_sigma = self.seq_len / 4
        window_kern_pos = torch.exp(-(dpos**2) / (2 * (window_sigma**2)))
        self.register_buffer("window_kern_pos", window_kern_pos)

        # Optional linear-time content branch.
        self.content_rank = int(content_rank)
        self.content_mix = float(content_mix)
        self.learned_content = bool(learned_content)
        if self.content_mix > 0.0:
            if self.learned_content:
                self.content_q_proj = nn.Linear(
                    self.embed_dim, self.num_heads * self.content_rank, bias=False
                )
                self.content_k_proj = nn.Linear(
                    self.embed_dim, self.num_heads * self.content_rank, bias=False
                )
            else:
                scale = 1.0 / math.sqrt(max(1, self.head_dim))
                gen = torch.Generator(device="cpu")
                seed = 1337 + (
                    self.embed_dim * 31
                    + self.seq_len * 17
                    + self.num_heads * 13
                    + self.content_rank * 7
                )
                gen.manual_seed(seed)
                self.register_buffer(
                    "content_wq",
                    torch.randn(
                        self.num_heads, self.head_dim, self.content_rank, generator=gen
                    )
                    * scale,
                    persistent=False,
                )
                self.register_buffer(
                    "content_wk",
                    torch.randn(
                        self.num_heads, self.head_dim, self.content_rank, generator=gen
                    )
                    * scale,
                    persistent=False,
                )

        self.use_ffn = bool(use_ffn)
        self.ffn_mult = int(ffn_mult)
        if self.use_ffn:
            hidden = self.embed_dim * self.ffn_mult
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )
            self.ln_attn = nn.LayerNorm(self.embed_dim)
            self.ln_ffn = nn.LayerNorm(self.embed_dim)

    def _pool_by_chunks(self, x: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        """Pool embeddings per chunk, then broadcast to all tokens in each chunk.

        This implements latent thought units: all tokens in a chunk share the
        same routing logits, creating coherent attention patterns.

        Args:
            x: [B, S, D] embeddings
            chunk_ids: [B, S] integer chunk IDs

        Returns:
            pooled: [B, S, D] with chunk-level pooling broadcast to tokens
        """
        B, S, D = x.shape
        device = x.device

        # Find max chunk ID to determine number of chunks
        # Use tensor operations to avoid graph breaks
        max_chunk = chunk_ids.max() + 1
        chunk_pooled = torch.zeros((B, max_chunk, D), device=device, dtype=x.dtype)
        chunk_counts = torch.zeros((B, max_chunk), device=device, dtype=x.dtype)

        # Sum embeddings per chunk
        chunk_ids_expanded = chunk_ids.unsqueeze(-1).expand(-1, -1, D)
        chunk_pooled.scatter_add_(1, chunk_ids_expanded, x)

        # Count tokens per chunk
        chunk_counts.scatter_add_(
            1, chunk_ids, torch.ones_like(chunk_ids, dtype=x.dtype)
        )

        # Average
        chunk_pooled = chunk_pooled / (chunk_counts.unsqueeze(-1) + 1e-8)

        # Broadcast back to tokens
        pooled = torch.gather(chunk_pooled, 1, chunk_ids_expanded)

        return pooled

    def _pool_chunks_to_chunk_embeddings(
        self, x: torch.Tensor, chunk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool embeddings per chunk, returning one embedding per chunk.

        Args:
            x: [B, S, D]
            chunk_ids: [B, S]

        Returns:
            chunk_pooled: [B, C, D] pooled mean per chunk
            chunk_counts: [B, C] counts per chunk (float)
        """
        B, S, D = x.shape
        device = x.device

        # Compute max_chunk in a compile-friendly way
        max_chunk = chunk_ids.max() + 1  # Keep as tensor

        # Initialize tensors with proper size
        chunk_pooled = torch.zeros((B, max_chunk, D), device=device, dtype=x.dtype)
        chunk_counts = torch.zeros((B, max_chunk), device=device, dtype=x.dtype)

        # Use scatter operations that are compile-friendly
        chunk_ids_expanded = chunk_ids.unsqueeze(-1).expand(-1, -1, D)  # [B,S,D]
        chunk_pooled.scatter_add_(1, chunk_ids_expanded, x)
        chunk_counts.scatter_add_(
            1, chunk_ids, torch.ones_like(chunk_ids, dtype=x.dtype)
        )

        # Avoid division by zero
        chunk_pooled = chunk_pooled / (chunk_counts.unsqueeze(-1) + 1e-8)

        # Convert max_chunk to int only when needed (outside compile)
        return chunk_pooled, chunk_counts

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _causal_linear_attn_vec(
        self,
        qf: torch.Tensor,  # [B,H,S,R]
        kf: torch.Tensor,  # [B,H,S,R]
        v: torch.Tensor,  # [B,H,S,Hd]
    ) -> torch.Tensor:
        """Fully vectorized causal linear attention using cumsum.

        This replaces the chunked loop for maximum speed during training.
        """
        if qf.shape != kf.shape:
            raise ValueError("qf and kf must have the same shape")
        B, H, S, R = qf.shape
        Hd = v.shape[-1]

        with torch.amp.autocast(device_type=v.device.type, enabled=False):
            q32 = qf.to(torch.float32)
            k32 = kf.to(torch.float32)
            v32 = v.to(torch.float32)

            # Compute S = \sum k_j v_j^T and Z = \sum k_j
            # kv: [B,H,S,R,Hd]
            kv = k32.unsqueeze(-1) * v32.unsqueeze(-2)
            s_state = torch.cumsum(kv, dim=2)
            z_state = torch.cumsum(k32, dim=2)

            # num = q^T S -> [B,H,S,Hd]
            num = (q32.unsqueeze(-2) @ s_state).squeeze(-2)
            # den = q^T Z -> [B,H,S]
            den = (q32 * z_state).sum(dim=-1, keepdim=True)

            out = num / (den + 1e-6)
            return out.to(v.dtype)

    def _render_attn_kernel_vec(self, weights_mod: torch.Tensor) -> torch.Tensor:
        """Render Toeplitz kernels for all heads and tokens.

        Args:
            weights_mod: [B, H, M] or [B, S, H, M] weights

        Returns:
            kern: [B, ..., 2S-1]
        """
        # Use float32 for kernel rendering to ensure wave precision
        with torch.amp.autocast(device_type=weights_mod.device.type, enabled=False):
            wm = weights_mod.to(torch.float32)
            H, M, W, _ = self.freqs.shape

            # Reshape based on input (batch routing vs per-token routing)
            orig_shape = wm.shape
            wm_flat = wm.view(-1, H, M)  # [N, H, M]

            pos = self.pos_kern.view(1, 1, 1, 1, -1, 2)
            f = self.freqs.view(1, H, M, W, 1, 2)

            # dots: [1, H, M, W, 2S-1]
            dots = (pos * f).sum(dim=-1)

            phase = 2 * math.pi * dots + self.phases.view(1, H, M, W, 1)
            waves = (
                torch.sin(phase) if self.use_sin_waves else triangle_wave(phase)
            ) * self.amps.view(1, H, M, W, 1)

            # masks: [H, M, 2S-1]
            masks = waves.sum(dim=3).squeeze(0)

            # out: [N, H, 2S-1]
            out = torch.einsum("nhm,hmk->nhk", wm_flat, masks)

            # Reshape back: [B, H, 2S-1] or [B, S, H, 2S-1]
            final_shape = list(orig_shape[:-1]) + [out.shape[-1]]
            return out.view(final_shape).to(weights_mod.dtype)

    def _toeplitz_causal_apply_fft_vec(
        self, kern_pos: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Standard causal convolution for batched heads.

        Args:
            kern_pos: [B, H, S]
            x: [B, H, S, C]

        Returns:
            [B, H, S, C]
        """
        B, H, S, C = x.shape
        n = 2 * S
        out_dtype = x.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            kp = F.pad(kern_pos, (0, n - S)).to(torch.float32)
            K = torch.fft.rfft(kp, n=n)  # [B, H, n/2+1]

            xt = x.transpose(2, 3)
            xp = F.pad(xt, (0, n - S)).to(torch.float32)
            X = torch.fft.rfft(xp, n=n)  # [B, H, C, n/2+1]

            Y = X * K.unsqueeze(2)
            yt = torch.fft.irfft(Y, n=n)[..., :S]
            return yt.transpose(2, 3).to(out_dtype)

    def forward(
        self, x: torch.Tensor, chunk_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass - fully vectorized over heads.

        Args:
            x: [B, S, D] embeddings
            chunk_ids: [B, S] optional chunk IDs for thought unit routing

        Returns:
            [B, S, D] output embeddings
        """
        B, S, D = x.shape
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}")

        v = (
            self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        )  # [B,H,S,Hd]

        # Chunk-level pooling
        if chunk_ids is not None:
            chunk_pooled, _ = self._pool_chunks_to_chunk_embeddings(
                x, chunk_ids
            )  # [B,C,D]
            # [B, C, H*M]
            logits_chunks = self.gatings(chunk_pooled).view(
                B, -1, self.num_heads, self.num_masks
            )
            # [B, S, H, M]
            logits = torch.gather(
                logits_chunks,
                1,
                chunk_ids.view(B, S, 1, 1).expand(
                    -1, -1, self.num_heads, self.num_masks
                ),
            )
            # Modulation coeff per chunk gathered to tokens: [B, S, H, R]
            coeff_chunks = self.mod_mlp(chunk_pooled).view(
                B, -1, self.num_heads, self.mod_rank
            )
            coeff = torch.gather(
                coeff_chunks,
                1,
                chunk_ids.view(B, S, 1, 1).expand(
                    -1, -1, self.num_heads, self.mod_rank
                ),
            )
        else:
            pooled = x.mean(dim=1)
            logits = self.gatings(pooled).view(B, self.num_heads, self.num_masks)
            coeff = self.mod_mlp(pooled).view(B, self.num_heads, self.mod_rank)

        qm = (
            torch.sigmoid(self.q_mod(x)).view(B, S, self.num_heads, 1).transpose(1, 2)
        )  # [B,H,S,1]
        km = (
            torch.sigmoid(self.k_mod(x)).view(B, S, self.num_heads, 1).transpose(1, 2)
        )  # [B,H,S,1]

        # Weights & Modulation scaling
        weights = topk_sparse_weights(logits, k=self.topk_masks, temperature=0.7)
        if coeff.ndim == 4:
            scale = torch.einsum("bshr,rhm->bshm", coeff, self.mod_basis)
        else:
            scale = torch.einsum("bhr,rhm->bhm", coeff, self.mod_basis)

        weights_mod = (weights * (1.0 + scale)).clamp_min(0.0)
        weights_mod = weights_mod / (weights_mod.sum(dim=-1, keepdim=True) + 1e-8)

        # Kernel Rendering
        kern = self._render_attn_kernel_vec(weights_mod)
        if kern.ndim == 4:  # [B, S, H, 2S-1]
            # Select causal kernel per token for non-Toeplitz case (fallback)
            # Here we just take the sequence mean to keep it Toeplitz for now.
            kern = kern.mean(dim=1)

        kern_pos = kern[
            :, :, (self.seq_len - 1) : (self.seq_len - 1 + self.seq_len)
        ]  # [B, H, S]

        # Apply Attention
        dens = torch.sigmoid(self.attn_alpha * kern_pos)
        dens = dens * self.window_kern_pos.view(1, 1, S).to(dens.dtype)

        stacked = torch.cat([km, km * v], dim=-1)  # [B, H, S, Hd+1]
        out_stacked = self._toeplitz_causal_apply_fft_vec(dens, stacked)

        head_ctx = out_stacked[..., 1:] / (out_stacked[..., :1] + 1e-8)

        # Optional content-aware branch
        if self.content_mix > 0.0:
            if self.learned_content:
                q_c = (
                    self.content_q_proj(x)
                    .view(B, S, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                k_c = (
                    self.content_k_proj(x)
                    .view(B, S, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                content_ctx = self._causal_linear_attn_vec(
                    self._phi(q_c), self._phi(k_c), v
                )
            else:
                xh = x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
                qf = self._phi(torch.einsum("bhsd,hdr->bhsr", xh, self.content_wq))
                kf = self._phi(torch.einsum("bhsd,hdr->bhsr", xh, self.content_wk))
                content_ctx = self._causal_linear_attn_vec(qf, kf, v)
            head_ctx = (
                1.0 - self.content_mix
            ) * head_ctx + self.content_mix * content_ctx

        head_ctx = qm * head_ctx + (1.0 - qm) * v
        out = self.out_proj(head_ctx.transpose(1, 2).reshape(B, S, D))

        if not self.use_ffn:
            return out + x

        x = self.ln_attn(x + out)
        x = self.ln_ffn(x + self.ffn(x))
        return x


class WaveCharLM(nn.Module):
    """Causal LM built from WaveDensityAttentionBlock layers.

    Supports latent chunk-level thought units for improved coherence and efficiency.
    Chunks are computed internally during forward pass and shared across layers.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        seq_len: int = 256,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 4,
        num_masks: int = 16,
        num_waves_per_mask: int = 8,
        topk_masks: int = 8,
        attn_alpha: float = 3.0,
        content_rank: int = 8,
        content_mix: float = 0.15,
        learned_content: bool = True,
        use_sin_waves: bool = True,
        ffn_mult: int = 4,
        tie_embeddings: bool = False,
        use_chunks: bool = True,
        chunk_size: int = 64,
        chunk_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.tie_embeddings = bool(tie_embeddings)
        self.use_chunks = bool(use_chunks)
        self.chunk_size = int(chunk_size)
        self.chunk_reg_weight = float(chunk_reg_weight)

        self.tok_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        self.blocks = nn.ModuleList(
            [
                WaveDensityAttentionBlock(
                    embed_dim=self.embed_dim,
                    seq_len=self.seq_len,
                    num_heads=num_heads,
                    num_masks=num_masks,
                    num_waves_per_mask=num_waves_per_mask,
                    topk_masks=topk_masks,
                    attn_alpha=attn_alpha,
                    content_rank=content_rank,
                    content_mix=content_mix,
                    learned_content=learned_content,
                    use_sin_waves=use_sin_waves,
                    ffn_mult=ffn_mult,
                    use_sampling=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        if self.tie_embeddings:
            self.head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        return_chunk_reg: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional chunk-level thought units.

        Args:
            idx: [B, S] token indices
            return_chunk_reg: if True, return (logits, chunk_reg_loss)

        Returns:
            logits: [B, S, V] or (logits, reg_loss) if return_chunk_reg=True
        """
        B, S = idx.shape
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}")

        # Compute chunk assignments (latent thought units)
        chunk_ids = None
        if self.use_chunks:
            chunk_ids = segment_into_chunks(
                seq_len=S,
                chunk_size=self.chunk_size,
                tokens=idx,
                tokenizer=None,  # Could pass tokenizer for punctuation-based chunking
                use_punctuation=False,  # Simple fixed-size chunks by default
            )

        x = self.tok_emb(idx) + self.pos_emb

        # Optional: compute chunk consistency regularizer
        chunk_reg_loss = torch.tensor(0.0, device=x.device)
        if self.chunk_reg_weight > 0.0 and chunk_ids is not None:
            # Vectorized chunk consistency regularization
            # Encourage similar representations within chunks
            B, S, D = x.shape

            # Create one-hot encoding of chunk assignments [B, S, C]
            max_chunk = chunk_ids.max() + 1
            chunk_onehot = torch.zeros(B, S, max_chunk, device=x.device, dtype=x.dtype)
            chunk_onehot.scatter_(2, chunk_ids.unsqueeze(-1), 1.0)

            # Compute chunk means [B, C, D]
            chunk_counts = chunk_onehot.sum(dim=1)  # [B, C]
            chunk_sums = torch.einsum("bsc,bsd->bcd", chunk_onehot, x)  # [B, C, D]
            chunk_means = chunk_sums / (chunk_counts.unsqueeze(-1) + 1e-8)  # [B, C, D]

            # Broadcast chunk means back to tokens [B, S, D]
            chunk_means_expanded = torch.einsum(
                "bsc,bcd->bsd", chunk_onehot, chunk_means
            )

            # Compute variance within each chunk
            diff = x - chunk_means_expanded  # [B, S, D]
            squared_diff = diff**2  # [B, S, D]

            # Weight by chunk membership and sum
            weighted_var = squared_diff * chunk_onehot.sum(
                dim=2, keepdim=True
            )  # [B, S, D]
            chunk_reg_loss = weighted_var.sum() / (chunk_onehot.sum() + 1e-8)

        for blk in self.blocks:
            x = blk(x, chunk_ids=chunk_ids)
        x = self.ln(x)
        logits = self.head(x)

        if return_chunk_reg:
            return logits, self.chunk_reg_weight * chunk_reg_loss
        return logits


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 120,
    temp: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed_text: str = "",
    device: str = "cuda",
) -> str:
    model.eval()
    seq_len = model.seq_len

    seed_tokens = (
        tokenizer.encode(seed_text, add_special_tokens=False) if seed_text else []
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = seed_tokens + prompt_tokens

    if len(tokens) >= seq_len:
        idx = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
    else:
        try:
            pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        except Exception:
            pad_id = 0
        padding = [pad_id] * (seq_len - len(tokens))
        idx = torch.tensor([padding + tokens], dtype=torch.long, device=device)

    generated: list[int] = []
    for _ in range(max_tokens):
        logits = model(idx)
        next_logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0 and generated:
            prev = torch.tensor(generated, device=next_logits.device, dtype=torch.long)
            next_logits[0, prev] = next_logits[0, prev] / float(repetition_penalty)

        next_logits = next_logits / max(float(temp), 1e-6)

        if top_k and top_k > 0:
            v, _ = torch.topk(next_logits, k=min(int(top_k), next_logits.shape[-1]))
            cutoff = v[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < cutoff,
                torch.full_like(next_logits, -float("inf")),
                next_logits,
            )

        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumprobs > float(top_p)
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            next_logits = torch.full_like(next_logits, -float("inf")).scatter(
                1, sorted_idx, sorted_logits
            )

        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx[:, 1:], nxt], dim=1)
        generated.append(int(nxt.item()))

    return tokenizer.decode(generated)


def build_streaming_dataset(
    tokenizer,
    seq_len: int | None = None,
    *,
    buffer_size: int = 10_000,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
) -> Iterator[int]:
    """Stream a C4-style text dataset's tokens indefinitely (no local data files).

    Args:
        seq_len: accepted for backwards compatibility; not used.
        buffer_size: shuffle buffer for streaming dataset.
        dataset_name: Hugging Face dataset name.
        dataset_config: dataset config/subset (C4 default: en).
        split: dataset split.
    """
    from datasets import load_dataset

    def _load_split(name: str, config: str, split_name: str):
        try:
            return load_dataset(name, config, split=split_name, streaming=True)
        except Exception:
            for alt in ("train", "validation", "test"):
                if alt == split_name:
                    continue
                try:
                    return load_dataset(name, config, split=alt, streaming=True)
                except Exception:
                    pass
            raise

    dataset = _load_split(str(dataset_name), str(dataset_config), str(split))

    def token_stream() -> Iterator[int]:
        while True:
            shuffled = dataset.shuffle(buffer_size=buffer_size, seed=None)
            for example in shuffled:
                text = example.get("text", "")
                tokens = tokenizer.encode(text, add_special_tokens=False)
                for tok in tokens:
                    yield int(tok)

    return token_stream()


def build_streaming_mixed_dataset(
    tokenizer,
    *,
    buffer_size: int = 10_000,
    c4_dataset_name: str = "allenai/c4",
    c4_dataset_config: str = "en",
    c4_split: str = "train",
    ultrachat_dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    ultrachat_split: str = "train_sft",
    assistant_only_loss: bool = True,
    include_assistant_prefix_in_loss: bool = False,
    ultrachat_prob: float = 0.5,
    chunk_tokens: int = 2048,
) -> Iterator[tuple[int, int]]:
    """Stream a mixture of C4 and UltraChat.

    Produces a unified stream of (token_id, loss_mask). C4 tokens always have
    loss_mask=1. UltraChat tokens carry their turn-level mask.
    """

    if not (0.0 <= float(ultrachat_prob) <= 1.0):
        raise ValueError("ultrachat_prob must be in [0, 1]")
    chunk_tokens = max(int(chunk_tokens), 1)

    c4_stream = build_streaming_dataset(
        tokenizer,
        buffer_size=buffer_size,
        dataset_name=c4_dataset_name,
        dataset_config=c4_dataset_config,
        split=c4_split,
    )
    ultrachat_stream = build_streaming_ultrachat_dataset(
        tokenizer,
        split=ultrachat_split,
        buffer_size=buffer_size,
        dataset_name=ultrachat_dataset_name,
        assistant_only_loss=assistant_only_loss,
        include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
    )

    def token_stream() -> Iterator[tuple[int, int]]:
        while True:
            use_ultra = bool(
                torch.rand((), device="cpu").item() < float(ultrachat_prob)
            )
            if use_ultra:
                for _ in range(chunk_tokens):
                    yield next(ultrachat_stream)
            else:
                for _ in range(chunk_tokens):
                    yield int(next(c4_stream)), 1

    return token_stream()


def build_streaming_ultrachat_dataset(
    tokenizer,
    *,
    split: str = "train_sft",
    buffer_size: int = 10_000,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    assistant_only_loss: bool = True,
    include_assistant_prefix_in_loss: bool = False,
) -> Iterator[tuple[int, int]]:
    """Stream UltraChat as (token_id, loss_mask) tuples."""
    from datasets import load_dataset

    def _load_split(name: str, split_name: str):
        try:
            return load_dataset(name, split=split_name, streaming=True)
        except Exception:
            for alt in ("train", "validation", "test"):
                try:
                    return load_dataset(name, split=alt, streaming=True)
                except Exception:
                    pass
            raise

    dataset = _load_split(dataset_name, split)

    def _normalize_role(role: str) -> str:
        r = (role or "").strip().lower()
        if r in ("user", "human"):
            return "user"
        if r in ("assistant", "gpt", "bot"):
            return "assistant"
        if r == "system":
            return "system"
        return r or "user"

    def _extract_messages(example) -> list[dict]:
        msgs = example.get("messages")
        if isinstance(msgs, list) and msgs:
            return msgs
        if "prompt" in example and ("response" in example or "completion" in example):
            return [
                {"role": "user", "content": example.get("prompt", "")},
                {
                    "role": "assistant",
                    "content": example.get("response", example.get("completion", "")),
                },
            ]
        if "instruction" in example and ("output" in example or "response" in example):
            return [
                {"role": "user", "content": example.get("instruction", "")},
                {
                    "role": "assistant",
                    "content": example.get("output", example.get("response", "")),
                },
            ]
        if "inputs" in example and "targets" in example:
            return [
                {"role": "user", "content": example.get("inputs", "")},
                {"role": "assistant", "content": example.get("targets", "")},
            ]
        return []

    def _encode_turn(
        prefix: str, content: str, loss_on_content: bool, loss_on_prefix: bool
    ):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = (
            tokenizer.encode(content, add_special_tokens=False) if content else []
        )
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        ids = prefix_ids + content_ids + newline_ids
        mask = (
            ([1 if loss_on_prefix else 0] * len(prefix_ids))
            + ([1 if loss_on_content else 0] * len(content_ids))
            + ([0] * len(newline_ids))
        )
        return ids, mask

    def token_stream() -> Iterator[tuple[int, int]]:
        while True:
            shuffled = dataset.shuffle(buffer_size=buffer_size, seed=None)
            for example in shuffled:
                messages = _extract_messages(example)
                if not messages:
                    continue

                ids_all: list[int] = []
                mask_all: list[int] = []

                for msg in messages:
                    role = _normalize_role(msg.get("role") or msg.get("from") or "user")
                    content = (
                        msg.get("content") or msg.get("value") or msg.get("text") or ""
                    )
                    content = str(content).strip()
                    if not content:
                        continue

                    if role == "assistant":
                        prefix = "Assistant: "
                        loss_on_prefix = include_assistant_prefix_in_loss
                        loss_on_content = True
                    elif role == "system":
                        prefix = "System: "
                        loss_on_prefix = False
                        loss_on_content = False
                    else:
                        prefix = "User: "
                        loss_on_prefix = False
                        loss_on_content = False

                    ids, mask = _encode_turn(
                        prefix,
                        content,
                        loss_on_content=loss_on_content,
                        loss_on_prefix=loss_on_prefix,
                    )
                    ids_all.extend(ids)
                    mask_all.extend(mask)

                if not ids_all:
                    continue

                if not assistant_only_loss:
                    mask_all = [1] * len(mask_all)

                for tid, m in zip(ids_all, mask_all):
                    yield int(tid), int(m)

    return token_stream()


def sample_batch(
    stream,
    batch_size: int,
    seq_len: int,
    device: str,
    assistant_only_loss: bool = False,
    min_supervised_tokens: int = 8,
    max_resample_tries: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch for next-token prediction.

    Supports two stream types:
      - int token stream -> standard labels
      - (token, mask) stream -> masked labels use ignore_index=-100
    """
    x_batch: list[list[int]] = []
    y_batch: list[list[int]] = []

    for _ in range(int(batch_size)):
        for t in range(int(max_resample_tries)):
            items = [next(stream) for _ in range(int(seq_len) + 1)]
            if isinstance(items[0], (tuple, list)):
                toks = [int(tok) for (tok, _m) in items]
                masks = [int(_m) for (_tok, _m) in items]
                supervised = sum(masks[1:])

                if (
                    (not assistant_only_loss)
                    or supervised >= int(min_supervised_tokens)
                    or t == int(max_resample_tries) - 1
                ):
                    x_batch.append(toks[:-1])
                    if assistant_only_loss:
                        y = [tok if m else -100 for tok, m in zip(toks[1:], masks[1:])]
                    else:
                        y = toks[1:]
                    y_batch.append(y)
                    break
            else:
                toks = [int(tok) for tok in items]
                x_batch.append(toks[:-1])
                y_batch.append(toks[1:])
                break

    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x, y


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _strip_compile_prefix(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


@dataclass
class TrainConfig:
    seq_len: int = 256
    steps: int = 10_000
    batch_size: int = 32
    lr: float = 2e-4
    seed: int = 42
    warmup_steps: int = 500
    clip_grad: float = 0.5
    val_every: int = 100
    val_batches: int = 10
    save_every: int = 500
    checkpoint_path: str = "wda-130m-mom.pt"
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    use_bf16: bool = True


def train_streaming_lm(
    *,
    device: str,
    cfg: TrainConfig,
    model_cfg: dict,
    data_cfg: dict,
) -> None:
    import os
    import time
    from transformers import GPT2TokenizerFast

    torch.manual_seed(int(cfg.seed))

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset_kind = str(data_cfg.get("dataset", "ultrachat")).lower()
    buffer_size = int(data_cfg.get("buffer_size", 10_000))
    _aol = data_cfg.get("assistant_only_loss", None)
    if _aol is None:
        assistant_only_loss = dataset_kind in ("ultrachat", "flan", "mix")
    else:
        assistant_only_loss = bool(_aol)
    include_assistant_prefix_in_loss = bool(
        data_cfg.get("include_assistant_prefix_in_loss", False)
    )

    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split")
    val_split = data_cfg.get("val_split")

    if dataset_kind == "ultrachat":
        dataset_name = dataset_name or "HuggingFaceH4/ultrachat_200k"
        split = split or "train_sft"
        val_split = val_split or "test_sft"
    elif dataset_kind == "flan":
        dataset_name = dataset_name or "SirNeural/flan_v2"
        split = split or "train"
        val_split = val_split or "validation"
    elif dataset_kind == "c4":
        dataset_name = dataset_name or "allenai/c4"
        split = split or "train"
        val_split = val_split or "validation"
    elif dataset_kind == "mix":
        dataset_name = dataset_name or "HuggingFaceH4/ultrachat_200k"
        split = split or "train_sft"
        val_split = val_split or "test_sft"
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    if dataset_kind == "ultrachat" or dataset_kind == "flan":
        train_stream = build_streaming_ultrachat_dataset(
            tokenizer,
            split=str(split),
            buffer_size=buffer_size,
            dataset_name=str(dataset_name),
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
        )
        val_stream = build_streaming_ultrachat_dataset(
            tokenizer,
            split=str(val_split),
            buffer_size=buffer_size,
            dataset_name=str(dataset_name),
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
        )
    elif dataset_kind == "c4":
        dataset_config = str(data_cfg.get("dataset_config", "en"))
        train_stream = build_streaming_dataset(
            tokenizer,
            buffer_size=buffer_size,
            dataset_name=str(dataset_name),
            dataset_config=dataset_config,
            split=str(split),
        )
        val_stream = build_streaming_dataset(
            tokenizer,
            buffer_size=buffer_size,
            dataset_name=str(dataset_name),
            dataset_config=dataset_config,
            split=str(val_split),
        )
    else:  # mix
        train_stream = build_streaming_mixed_dataset(
            tokenizer,
            buffer_size=buffer_size,
            c4_dataset_name=str(data_cfg.get("c4_dataset_name", "allenai/c4")),
            c4_dataset_config=str(data_cfg.get("c4_dataset_config", "en")),
            c4_split=str(data_cfg.get("c4_split", "train")),
            ultrachat_dataset_name=str(
                data_cfg.get("ultrachat_dataset_name", dataset_name)
            ),
            ultrachat_split=str(data_cfg.get("ultrachat_split", split)),
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
            ultrachat_prob=float(data_cfg.get("ultrachat_prob", 0.5)),
            chunk_tokens=int(data_cfg.get("mix_chunk_tokens", 2048)),
        )
        val_stream = build_streaming_mixed_dataset(
            tokenizer,
            buffer_size=buffer_size,
            c4_dataset_name=str(data_cfg.get("c4_dataset_name", "allenai/c4")),
            c4_dataset_config=str(data_cfg.get("c4_dataset_config", "en")),
            c4_split=str(data_cfg.get("c4_val_split", "validation")),
            ultrachat_dataset_name=str(
                data_cfg.get("ultrachat_dataset_name", dataset_name)
            ),
            ultrachat_split=str(data_cfg.get("ultrachat_val_split", val_split)),
            assistant_only_loss=assistant_only_loss,
            include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
            ultrachat_prob=float(data_cfg.get("ultrachat_prob", 0.5)),
            chunk_tokens=int(data_cfg.get("mix_chunk_tokens", 2048)),
        )

    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=int(cfg.seq_len),
        embed_dim=int(model_cfg.get("embed_dim", 768)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_masks=int(model_cfg.get("num_masks", 16)),
        num_waves_per_mask=int(model_cfg.get("num_waves_per_mask", 8)),
        topk_masks=int(model_cfg.get("topk_masks", 8)),
        attn_alpha=float(model_cfg.get("attn_alpha", 3.0)),
        content_rank=int(model_cfg.get("content_rank", 8)),
        content_mix=float(model_cfg.get("content_mix", 0.15)),
        learned_content=bool(model_cfg.get("learned_content", True)),
        use_sin_waves=bool(model_cfg.get("use_sin_waves", True)),
        ffn_mult=int(model_cfg.get("ffn_mult", 4)),
        tie_embeddings=bool(model_cfg.get("tie_embeddings", False)),
        use_chunks=bool(model_cfg.get("use_chunks", True)),
        chunk_size=int(model_cfg.get("chunk_size", 64)),
        chunk_reg_weight=float(model_cfg.get("chunk_reg_weight", 0.0)),
    ).to(device)

    if device.startswith("cuda") and bool(cfg.use_compile):
        try:
            model = torch.compile(model, mode=str(cfg.compile_mode), fullgraph=False)  # type: ignore
            print(f"✅ torch.compile enabled (mode={cfg.compile_mode})")
        except Exception as e:
            print(f"⚠️ torch.compile unavailable, continuing without it: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step < int(cfg.warmup_steps):
            return float(cfg.lr) * (step / max(int(cfg.warmup_steps), 1))
        return float(cfg.lr)

    use_amp = device.startswith("cuda")
    if device.startswith("cuda"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        amp_device = "cuda"
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        amp_device = "cpu"
        use_amp = False

    start_step = 1
    best_val_loss = float("inf")

    if os.path.exists(cfg.checkpoint_path):
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        state = _strip_compile_prefix(ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        start_step = int(ckpt.get("step", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from step {start_step-1}")

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    step_times: list[float] = []
    start_time = time.time()
    last_step_time = start_time

    for step in range(start_step, int(cfg.steps) + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)

        x, y = sample_batch(
            train_stream,
            batch_size=int(cfg.batch_size),
            seq_len=int(cfg.seq_len),
            device=device,
            assistant_only_loss=assistant_only_loss,
        )

        opt.zero_grad(set_to_none=True)

        amp_dtype = torch.bfloat16 if bool(cfg.use_bf16) else torch.float16
        with torch.amp.autocast(
            device_type=amp_device, enabled=use_amp, dtype=amp_dtype
        ):
            # Use chunk regularization if enabled
            use_reg = (
                model.chunk_reg_weight > 0.0
                if hasattr(model, "chunk_reg_weight")
                else False
            )
            if use_reg:
                logits, reg_loss = model(x, return_chunk_reg=True)
            else:
                logits = model(x)
                reg_loss = torch.tensor(0.0, device=x.device)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
            )
            total_loss = loss + reg_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf loss at step {step}, skipping")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
        scaler.step(opt)
        scaler.update()

        now = time.time()
        step_times.append(now - last_step_time)
        if len(step_times) > 100:
            step_times.pop(0)
        last_step_time = now

        if step % int(cfg.val_every) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vlosses: list[float] = []
                for _ in range(int(cfg.val_batches)):
                    vx, vy = sample_batch(
                        val_stream,
                        batch_size=int(cfg.batch_size),
                        seq_len=int(cfg.seq_len),
                        device=device,
                        assistant_only_loss=assistant_only_loss,
                    )
                    with torch.amp.autocast(
                        device_type=amp_device, enabled=use_amp, dtype=amp_dtype
                    ):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(
                            vlogits.view(-1, vlogits.size(-1)),
                            vy.view(-1),
                            ignore_index=-100,
                        )
                    vlosses.append(float(vloss.item()))
                vloss = float(sum(vlosses) / len(vlosses))

            model.train()

            if vloss < best_val_loss:
                best_val_loss = vloss
                best_path = cfg.checkpoint_path.replace(".pt", "_best.pt")
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "tokenizer": "gpt2",
                        "model_cfg": model_cfg,
                        "data_cfg": data_cfg,
                        "train_cfg": cfg.__dict__,
                    },
                    best_path,
                )

            avg_step_time = sum(step_times) / len(step_times)
            eta_s = avg_step_time * (int(cfg.steps) - step)
            ppl = math.exp(min(vloss, 10.0))
            reg_str = (
                f" | reg {reg_loss.item():.4f}"
                if use_reg and reg_loss.item() > 0
                else ""
            )
            print(
                f"[{step:5d}] train {loss.item():.4f}{reg_str} | val {vloss:.4f} | ppl {ppl:.2f} | "
                f"{avg_step_time*1000:.0f}ms/step | ETA {eta_s/60:.1f}m"
            )

        if step % int(cfg.save_every) == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "tokenizer": "gpt2",
                    "model_cfg": model_cfg,
                    "data_cfg": data_cfg,
                    "train_cfg": cfg.__dict__,
                },
                cfg.checkpoint_path,
            )


def _cmd_train(args: argparse.Namespace) -> None:
    device = args.device or _best_device()
    cfg = TrainConfig(
        seq_len=args.seq_len,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
    )

    model_cfg = {
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_masks": args.num_masks,
        "num_waves_per_mask": args.num_waves_per_mask,
        "topk_masks": args.topk_masks,
        "attn_alpha": args.attn_alpha,
        "content_rank": args.content_rank,
        "content_mix": args.content_mix,
        "learned_content": bool(args.learned_content),
        "use_sin_waves": bool(args.use_sin_waves),
        "ffn_mult": args.ffn_mult,
        "tie_embeddings": bool(args.tie_embeddings),
        "use_chunks": bool(args.use_chunks),
        "chunk_size": args.chunk_size,
        "chunk_reg_weight": args.chunk_reg_weight,
    }

    data_cfg = {
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "dataset_config": getattr(args, "dataset_config", "en"),
        "split": args.split,
        "val_split": args.val_split,
        "buffer_size": args.buffer_size,
        "assistant_only_loss": bool(args.assistant_only_loss),
        "include_assistant_prefix_in_loss": bool(args.include_assistant_prefix_in_loss),
        "ultrachat_prob": float(getattr(args, "mix_ultrachat_prob", 0.5)),
        "mix_chunk_tokens": int(getattr(args, "mix_chunk_tokens", 2048)),
        "c4_dataset_name": str(getattr(args, "c4_dataset_name", "allenai/c4")),
        "c4_dataset_config": str(getattr(args, "c4_dataset_config", "en")),
        "c4_split": str(getattr(args, "c4_split", "train")),
        "c4_val_split": str(getattr(args, "c4_val_split", "validation")),
        "ultrachat_dataset_name": str(
            getattr(args, "ultrachat_dataset_name", "HuggingFaceH4/ultrachat_200k")
        ),
        "ultrachat_split": str(getattr(args, "ultrachat_split", "train_sft")),
        "ultrachat_val_split": str(getattr(args, "ultrachat_val_split", "test_sft")),
    }

    print("Device:", device)
    train_streaming_lm(device=device, cfg=cfg, model_cfg=model_cfg, data_cfg=data_cfg)


def _sanitize_argv(argv: list[str]) -> list[str]:
    """Strip IPython/Jupyter kernel launcher args.

    When running this file via `%run model_train.py` or similar in a notebook,
    IPython may populate argv with the kernel connection file (and sometimes a
    preceding `-f`). That confuses our subcommand parser.
    """

    if not argv:
        return []

    # Common ipykernel forms:
    #   ['-f', '/path/to/kernel-xxxx.json']
    #   ['/path/to/kernel-xxxx.json']
    if len(argv) >= 2 and argv[0] == "-f" and argv[1].endswith(".json"):
        return argv[2:]
    if argv[0].endswith(".json"):
        return argv[1:]

    return argv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Wave-Density Attention (shippable reference)"
    )
    sub = ap.add_subparsers(dest="cmd")

    train = sub.add_parser(
        "train", help="Train a WDA LM from scratch (streaming datasets)"
    )
    train.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")

    train.add_argument(
        "--dataset", default="ultrachat", choices=["ultrachat", "c4", "mix", "flan"]
    )
    train.add_argument("--dataset-name", default=None, help="Override HF dataset name")
    train.add_argument(
        "--split", default=None, help="Train split (UltraChat default: train_sft)"
    )
    train.add_argument(
        "--val-split", default=None, help="Val split (UltraChat default: test_sft)"
    )
    train.add_argument(
        "--dataset-config",
        default="en",
        help="Dataset config/subset (C4 default: en). Ignored for UltraChat.",
    )
    train.add_argument(
        "--mix-ultrachat-prob",
        type=float,
        default=0.5,
        help="When --dataset=mix, probability of sampling UltraChat chunks (default: 0.5).",
    )
    train.add_argument(
        "--mix-chunk-tokens",
        type=int,
        default=2048,
        help="When --dataset=mix, tokens per chunk before switching sources.",
    )
    train.add_argument(
        "--c4-dataset-name",
        default="allenai/c4",
        help="When --dataset=mix, HF dataset name for the C4 stream.",
    )
    train.add_argument(
        "--c4-dataset-config",
        default="en",
        help="When --dataset=mix, dataset config/subset for the C4 stream.",
    )
    train.add_argument(
        "--c4-split",
        default="train",
        help="When --dataset=mix, split for the C4 training stream.",
    )
    train.add_argument(
        "--c4-val-split",
        default="validation",
        help="When --dataset=mix, split for the C4 validation stream.",
    )
    train.add_argument(
        "--ultrachat-dataset-name",
        default="HuggingFaceH4/ultrachat_200k",
        help="When --dataset=mix, HF dataset name for the UltraChat stream.",
    )
    train.add_argument(
        "--ultrachat-split",
        default="train_sft",
        help="When --dataset=mix, split for the UltraChat training stream.",
    )
    train.add_argument(
        "--ultrachat-val-split",
        default="test_sft",
        help="When --dataset=mix, split for the UltraChat validation stream.",
    )
    train.add_argument("--buffer-size", type=int, default=10_000)
    train.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If true, compute loss only on assistant tokens (default: true for ultrachat, false for c4)",
    )
    train.add_argument(
        "--include-assistant-prefix-in-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include the 'Assistant:' prefix tokens in the loss (UltraChat).",
    )

    train.add_argument("--seq-len", type=int, default=256)
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--checkpoint-path", default="wda-130m-mom.pt")

    train.add_argument("--embed-dim", type=int, default=768)
    train.add_argument("--num-layers", type=int, default=8)
    train.add_argument("--num-heads", type=int, default=4)
    train.add_argument("--num-masks", type=int, default=16)
    train.add_argument("--num-waves-per-mask", type=int, default=8)
    train.add_argument("--topk-masks", type=int, default=8)
    train.add_argument("--attn-alpha", type=float, default=3.0)
    train.add_argument("--content-rank", type=int, default=8)
    train.add_argument("--content-mix", type=float, default=0.15)
    train.add_argument(
        "--learned-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learned projections for the optional content-mix branch.",
    )
    train.add_argument(
        "--use-sin-waves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use sin() waves (faster) instead of triangle waves.",
    )
    train.add_argument("--ffn-mult", type=int, default=4)
    train.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie token embedding and output head weights.",
    )
    train.add_argument(
        "--use-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable chunk-level thought units (latent structural grouping).",
    )
    train.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Max tokens per thought unit chunk (default: 64).",
    )
    train.add_argument(
        "--chunk-reg-weight",
        type=float,
        default=0.0,
        help="Regularization weight for chunk consistency (0 = disabled).",
    )

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    clean_argv = _sanitize_argv(raw_argv)

    # In notebooks, calling `%run model_train.py` (with no args) should show help
    # rather than raising a SystemExit.
    if not clean_argv:
        ap.print_help()
        return

    args = ap.parse_args(clean_argv)

    if args.cmd == "train":
        if args.dataset_name is None:
            if args.dataset == "ultrachat" or args.dataset == "mix":
                args.dataset_name = "HuggingFaceH4/ultrachat_200k"
            elif args.dataset == "flan":
                args.dataset_name = "SirNeural/flan_v2"
            else:
                args.dataset_name = "allenai/c4"
        if args.split is None:
            if args.dataset == "ultrachat" or args.dataset == "mix":
                args.split = "train_sft"
            elif args.dataset == "flan":
                args.split = "train"
            else:
                args.split = "train"
        if args.val_split is None:
            if args.dataset == "ultrachat" or args.dataset == "mix":
                args.val_split = "test_sft"
            elif args.dataset == "flan":
                args.val_split = "validation"
            else:
                args.val_split = "validation"
        if args.assistant_only_loss is None:
            args.assistant_only_loss = args.dataset in ("ultrachat", "mix", "flan")
        _cmd_train(args)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
