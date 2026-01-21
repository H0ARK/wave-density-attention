import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Selective disable for torch.fft in compile if needed
try:
    from torch._dynamo import disable as dynamo_disable
except Exception:

    def dynamo_disable(fn):
        return fn


class StraightThroughHeaviside(nn.Module):
    """Binary step with a smooth surrogate gradient."""

    def __init__(self, alpha: float = 6.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hard = (x > 0).to(x.dtype)
        y_soft = torch.sigmoid(self.alpha * x)
        return y_hard + (y_soft - y_hard).detach()


def triangle_wave(x: torch.Tensor) -> torch.Tensor:
    """Fast, stable triangle wave implementation in [-1, 1]."""
    t = (x - (math.pi / 2)) / (2 * math.pi)
    frac = t - torch.floor(t)
    tri01 = 1.0 - 2.0 * torch.abs(frac - 0.5)
    return 2.0 * tri01 - 1.0


def topk_sparse_weights(
    logits: torch.Tensor, k: int, temperature: float = 1.0
) -> torch.Tensor:
    """Return a sparse weight vector with exactly top-k nonzeros per row."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    soft = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    topk_idx = torch.topk(soft, k=min(k, soft.shape[-1]), dim=-1).indices
    hard = torch.zeros_like(soft)
    hard.scatter_(dim=-1, index=topk_idx, value=1.0)
    hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-8)
    return hard + (soft - hard).detach()


class WaveDensityBlock(nn.Module):
    """Wave-interference -> density map block for 1D/2D grid."""

    def __init__(
        self,
        embed_dim: int = 64,
        seq_len: int = 64,
        num_masks: int = 32,
        num_waves_per_mask: int = 16,
        num_heads: int = 4,
        num_samples: int = 128,
        noise_sigma: float = 0.12,
        topk_masks: int = 8,
        gate_hidden: int = 128,
        step_alpha: float = 6.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.seq_len = int(seq_len)
        self.grid_size = int(math.sqrt(self.seq_len))
        self.num_masks = int(num_masks)
        self.num_waves_per_mask = int(num_waves_per_mask)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_samples = int(num_samples)
        self.noise_sigma = float(noise_sigma)
        self.topk_masks = int(topk_masks)

        self.gatings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, self.num_masks),
                )
                for _ in range(self.num_heads)
            ]
        )

        self.freqs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask, 2) * 4.0
                )
                for _ in range(self.num_heads)
            ]
        )
        self.amps = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask) * 0.05 + 0.8
                )
                for _ in range(self.num_heads)
            ]
        )
        self.phases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask) * 2 * math.pi
                )
                for _ in range(self.num_heads)
            ]
        )

        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self.register_buffer("pos", pos)

        self.step = StraightThroughHeaviside(alpha=step_alpha)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _waves_to_grid(self, head: int) -> torch.Tensor:
        pos = self.pos.unsqueeze(0).unsqueeze(0)
        freqs = self.freqs[head].unsqueeze(-2)
        dots = torch.sum(pos * freqs, dim=-1)
        phases = self.phases[head].unsqueeze(-1)
        waves = triangle_wave(2 * math.pi * dots + phases)
        waves = waves * self.amps[head].unsqueeze(-1)
        mask_grids = waves.sum(dim=1)
        return mask_grids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        pooled = x.mean(dim=1)
        head_outputs = []
        for h in range(self.num_heads):
            logits = self.gatings[h](pooled)
            weights = topk_sparse_weights(logits, k=self.topk_masks, temperature=0.6)
            mask_grids = self._waves_to_grid(h)
            final_grid = torch.matmul(weights, mask_grids)
            noise = (
                torch.randn(B, self.num_samples, S, device=x.device, dtype=x.dtype)
                * self.noise_sigma
            )
            samples = final_grid.unsqueeze(1) + noise
            hits = self.step(samples)
            density = hits.mean(dim=1)
            density = density / (density.sum(dim=1, keepdim=True) + 1e-8)
            head_x = x[:, :, h * self.head_dim : (h + 1) * self.head_dim]
            head_out = density.unsqueeze(-1) * head_x
            head_outputs.append(head_out)
        multihead_out = torch.cat(head_outputs, dim=-1)
        return self.out_proj(multihead_out) + x


class WaveDensityAttentionBlock(nn.Module):
    """Causal self-attention where attention logits are generated by wave interference."""

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 4,
        num_masks: int = 128,
        num_waves_per_mask: int = 32,
        topk_masks: int = 8,
        gate_hidden: int = 128,
        gate_temp: float = 1.0,
        attn_alpha: float = 3.0,
        content_rank: int = 8,
        content_mix: float = 0.0,
        learned_content: bool = False,
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
        self.num_masks = int(num_masks)
        self.num_waves_per_mask = int(num_waves_per_mask)
        self.topk_masks = int(topk_masks)
        self.gate_temp = float(gate_temp)
        self.attn_alpha = float(attn_alpha)
        self.use_sampling = bool(use_sampling)
        self.num_samples = int(num_samples)
        self.noise_sigma = float(noise_sigma)
        self.step = StraightThroughHeaviside(alpha=step_alpha)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_sin_waves = bool(use_sin_waves)

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Metrics storage (not persistent)
        self.routing_stats = {}

        self.content_rank = int(content_rank)
        self.content_mix = float(content_mix)
        self.learned_content = bool(learned_content)
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
            torch.randn(self.num_heads, self.head_dim, self.content_rank, generator=gen)
            * scale,
            persistent=False,
        )
        self.register_buffer(
            "content_wk",
            torch.randn(self.num_heads, self.head_dim, self.content_rank, generator=gen)
            * scale,
            persistent=False,
        )
        if self.learned_content:
            self.content_q_proj = nn.Linear(
                self.embed_dim, self.num_heads * self.content_rank, bias=False
            )
            self.content_k_proj = nn.Linear(
                self.embed_dim, self.num_heads * self.content_rank, bias=False
            )

        self.q_mod = nn.Linear(self.embed_dim, self.num_heads)
        self.k_mod = nn.Linear(self.embed_dim, self.num_heads)

        self.gatings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, self.num_masks),
                )
                for _ in range(self.num_heads)
            ]
        )

        self.mod_rank = 8
        self.mod_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, self.mod_rank),
                )
                for _ in range(self.num_heads)
            ]
        )
        self.mod_basis = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.mod_rank, self.num_masks) * 0.05)
                for _ in range(self.num_heads)
            ]
        )

        self.freqs = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask, 2) * 1.5
                )
                for _ in range(self.num_heads)
            ]
        )
        self.amps = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask) * 0.02 + 0.5
                )
                for _ in range(self.num_heads)
            ]
        )
        self.phases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_masks, self.num_waves_per_mask) * math.pi
                )
                for _ in range(self.num_heads)
            ]
        )

        rel_dist_kern = torch.arange(-(self.seq_len - 1), self.seq_len).float()
        p1_k = rel_dist_kern / self.seq_len
        p2_k = (
            torch.sign(rel_dist_kern)
            * torch.sqrt(torch.abs(rel_dist_kern) + 1e-6)
            / math.sqrt(self.seq_len)
        )
        self.register_buffer("pos_kern", torch.stack([p1_k, p2_k], dim=-1))

        dpos = torch.arange(0, self.seq_len).float()
        window_sigma = self.seq_len / 4
        self.register_buffer(
            "window_kern_pos", torch.exp(-(dpos**2) / (2 * (window_sigma**2)))
        )

        self.use_ffn = use_ffn
        self.ffn_mult = int(ffn_mult)
        self.collect_stats = True
        self.cache_freeze_routing = True
        self.cache_disable_content = True
        if self.use_ffn:
            hidden = self.embed_dim * self.ffn_mult
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )
            self.ln_attn = nn.LayerNorm(self.embed_dim)
            self.ln_ffn = nn.LayerNorm(self.embed_dim)

        # Optimization: Pre-rendered mask cache for inference
        self.register_buffer("_mask_cache", None, persistent=False)

    def _render_all_masks(self):
        """Pre-renders all static masks to a cache [H, M, 2S-1]."""
        with torch.no_grad():
            rendered_heads = []
            for h in range(self.num_heads):
                # Prepare pos_kern [1, 2S-1, 2]
                pos = self.pos_kern.unsqueeze(0)
                # Prepare params [M, 1, 2], [M, 1], [M, 1]
                freqs = self.freqs[h].unsqueeze(-2)
                phases = self.phases[h].unsqueeze(-1)
                amps = self.amps[h].unsqueeze(-1)

                # Render waves (matching _render_attn_kernel logic)
                dots = torch.sum(pos * freqs, dim=-1)
                phase = 2 * math.pi * dots + phases
                waves = (
                    torch.sin(phase) if self.use_sin_waves else triangle_wave(phase)
                ) * amps

                # Sum waves for each mask -> [M, 2S-1]
                mask_kernels = waves.sum(dim=1)
                rendered_heads.append(mask_kernels)

            # Stack all heads -> [H, M, 2S-1]
            self._mask_cache = torch.stack(rendered_heads)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _causal_linear_attn_chunked(self, qf, kf, v, chunk_size=64):
        B, H, S, R = qf.shape
        Hd = v.shape[-1]
        out = torch.empty((B, H, S, Hd), device=v.device, dtype=v.dtype)
        eps = 1e-6
        S_state = torch.zeros((B, H, R, Hd), device=v.device, dtype=torch.float32)
        Z_state = torch.zeros((B, H, R), device=v.device, dtype=torch.float32)
        with torch.amp.autocast(device_type=v.device.type, enabled=False):
            qf32, kf32, v32 = (
                qf.to(torch.float32),
                kf.to(torch.float32),
                v.to(torch.float32),
            )
            for start in range(0, S, chunk_size):
                end = min(S, start + chunk_size)
                q, k, vc = (
                    qf32[:, :, start:end, :],
                    kf32[:, :, start:end, :],
                    v32[:, :, start:end, :],
                )
                kv = k.unsqueeze(-1) * vc.unsqueeze(-2)
                kv_prefix = torch.cumsum(kv, dim=2)
                k_prefix = torch.cumsum(k, dim=2)
                S_chunk = S_state.unsqueeze(2) + kv_prefix
                Z_chunk = Z_state.unsqueeze(2) + k_prefix
                num = (q.unsqueeze(-1) * S_chunk).sum(dim=-2)
                den = (q * Z_chunk).sum(dim=-1)
                out[:, :, start:end, :] = (num / (den.unsqueeze(-1) + eps)).to(
                    out.dtype
                )
                S_state = S_state + kv_prefix[:, :, -1, :, :]
                Z_state = Z_state + k_prefix[:, :, -1, :]
        return out

    @dynamo_disable
    def _render_attn_kernel(self, head, weights_mod):
        # Optimization: Use mask cache in evaluation mode
        if not self.training:
            if self._mask_cache is None:
                self._render_all_masks()
            # weights_mod: [B, M]
            # _mask_cache[head]: [M, 2S-1]
            # Check for device mismatch
            cache = self._mask_cache[head]
            if cache.device != weights_mod.device:
                cache = cache.to(weights_mod.device)
            return torch.matmul(weights_mod, cache.to(weights_mod.dtype))

        B, M = weights_mod.shape
        out_dtype = weights_mod.dtype
        with torch.amp.autocast(device_type=weights_mod.device.type, enabled=False):
            wm = weights_mod.to(torch.float32)
            mask_sums = wm.sum(dim=0)
            active_mask_indices = torch.nonzero(mask_sums > 0).squeeze(-1)
            if active_mask_indices.numel() == 0:
                return torch.zeros(
                    B, 2 * self.seq_len - 1, device=weights_mod.device, dtype=out_dtype
                )
            pos = self.pos_kern.unsqueeze(0).unsqueeze(0)
            freqs = self.freqs[head][active_mask_indices].unsqueeze(-2)
            phases = self.phases[head][active_mask_indices].unsqueeze(-1)
            amps = self.amps[head][active_mask_indices].unsqueeze(-1)
            dots = torch.sum(pos * freqs, dim=-1)
            phase = 2 * math.pi * dots + phases
            waves = (
                torch.sin(phase) if self.use_sin_waves else triangle_wave(phase)
            ) * amps
            surf_kerns = waves.sum(dim=1)
            surf_kerns = surf_kerns.to(wm.dtype)
            attn_kern_batch = torch.matmul(wm[:, active_mask_indices], surf_kerns)
            return attn_kern_batch.to(out_dtype)

    @dynamo_disable
    def _toeplitz_causal_apply_fft_stacked(self, kern_pos, x):
        B, S, C = x.shape
        n = 2 * S
        out_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            kp = F.pad(kern_pos, (0, n - S)).to(torch.float32)
            K = torch.fft.rfft(kp, n=n)
            xt = x.transpose(1, 2)
            xp = F.pad(xt, (0, n - S)).to(torch.float32)
            X = torch.fft.rfft(xp, n=n)
            Y = X * K.unsqueeze(1)
            yt = torch.fft.irfft(Y, n=n)[..., :S]
            y = yt.transpose(1, 2)
            return y.to(out_dtype)

    def _build_kernels_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        kernels = []
        for h in range(self.num_heads):
            logits = self.gatings[h](pooled)
            weights = topk_sparse_weights(
                logits, k=self.topk_masks, temperature=self.gate_temp
            )
            coeff = self.mod_mlp[h](pooled)
            scale = 0.5 * torch.tanh(
                torch.einsum("br,rm->bm", coeff, self.mod_basis[h])
            )
            weights_mod = (weights * (1.0 + scale)).clamp_min(0.0)
            weights_mod = weights_mod / (weights_mod.sum(dim=-1, keepdim=True) + 1e-8)
            weights_mod = weights_mod.to(dtype=pooled.dtype)

            attn_kern_batch = self._render_attn_kernel(h, weights_mod)
            mid = self.seq_len - 1
            attn_kern_pos = attn_kern_batch[:, mid : mid + self.seq_len]

            # Deterministic kernel for cache (avoid sampling noise in cache path)
            dens_kern_pos = torch.sigmoid(self.attn_alpha * attn_kern_pos)
            dens_kern_pos = dens_kern_pos * self.window_kern_pos[
                : self.seq_len
            ].unsqueeze(0).to(dens_kern_pos.dtype)
            kernels.append(dens_kern_pos)
        return torch.stack(kernels, dim=1)

    def _init_wda_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B, S, _ = x.shape
        L = min(S, self.seq_len)
        x_cache = x[:, -L:, :]
        v = self.v_proj(x_cache).view(B, L, self.num_heads, self.head_dim)
        km = torch.sigmoid(self.k_mod(x_cache))
        pooled = x_cache.mean(dim=1)
        kernel = self._build_kernels_from_pooled(pooled)

        cache_len = torch.full((B,), L, device=x.device, dtype=torch.int32)

        content_S = None
        content_Z = None
        if self.content_mix > 0.0 and not self.cache_disable_content:
            if self.learned_content:
                q = (
                    self.content_q_proj(x_cache)
                    .view(B, L, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                k = (
                    self.content_k_proj(x_cache)
                    .view(B, L, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
            else:
                xh = x_cache.reshape(B, L, self.num_heads, self.head_dim).transpose(
                    1, 2
                )
                q = torch.einsum("bhsd,hdr->bhsr", xh, self.content_wq)
                k = torch.einsum("bhsd,hdr->bhsr", xh, self.content_wk)

            qf = self._phi(q)
            kf = self._phi(k)
            vh = v.transpose(1, 2)  # B, H, L, Hd
            content_S = (kf.unsqueeze(-1) * vh.unsqueeze(-2)).sum(dim=2)
            content_Z = kf.sum(dim=2)

        return (x_cache, km, v, kernel, cache_len, content_S, content_Z)

    def _forward_wda_cached(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        B, S, D = x.shape
        if S != 1:
            # Cached path expects one token at a time.
            out = self.forward(x, past_key_value=None, use_cache=False)
            cache = self._init_wda_cache(x)
            return out, cache

        if past_key_value is None:
            cache = self._init_wda_cache(x)
            x_cache, km_cache, v_cache, kernel, cache_len, content_S, content_Z = cache
        else:
            x_cache, km_cache, v_cache, kernel, cache_len, content_S, content_Z = (
                past_key_value
            )

        # Append new token
        v_new = self.v_proj(x).view(B, 1, self.num_heads, self.head_dim)
        km_new = torch.sigmoid(self.k_mod(x))
        qm_new = torch.sigmoid(self.q_mod(x))

        x_cache = torch.cat([x_cache, x], dim=1)
        km_cache = torch.cat([km_cache, km_new], dim=1)
        v_cache = torch.cat([v_cache, v_new], dim=1)

        if x_cache.size(1) > self.seq_len:
            x_cache = x_cache[:, -self.seq_len :, :]
            km_cache = km_cache[:, -self.seq_len :, :]
            v_cache = v_cache[:, -self.seq_len :, :, :]

        L = x_cache.size(1)
        cache_len = torch.full((B,), L, device=x.device, dtype=torch.int32)

        if not self.cache_freeze_routing:
            pooled = x_cache.mean(dim=1)
            kernel = self._build_kernels_from_pooled(pooled)

        content_ctx = None
        if self.content_mix > 0.0 and not self.cache_disable_content:
            if self.learned_content:
                q = (
                    self.content_q_proj(x)
                    .view(B, 1, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                k = (
                    self.content_k_proj(x)
                    .view(B, 1, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
            else:
                xh = x.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
                q = torch.einsum("bhsd,hdr->bhsr", xh, self.content_wq)
                k = torch.einsum("bhsd,hdr->bhsr", xh, self.content_wk)

            qf = self._phi(q)
            kf = self._phi(k)
            vh_new = v_new.transpose(1, 2)  # B, H, 1, Hd
            if content_S is None or content_Z is None:
                content_S = torch.zeros(
                    (B, self.num_heads, self.content_rank, self.head_dim),
                    device=x.device,
                    dtype=torch.float32,
                )
                content_Z = torch.zeros(
                    (B, self.num_heads, self.content_rank),
                    device=x.device,
                    dtype=torch.float32,
                )
            content_S = content_S + (kf.unsqueeze(-1) * vh_new.unsqueeze(-2)).sum(dim=2)
            content_Z = content_Z + kf.sum(dim=2)
            num = (qf.unsqueeze(-1) * content_S.unsqueeze(2)).sum(dim=-2)
            den = (qf * content_Z.unsqueeze(2)).sum(dim=-1)
            content_ctx = (num / (den.unsqueeze(-1) + 1e-6)).to(v_new.dtype)
            content_ctx = content_ctx[:, :, 0, :]

        out_heads = []
        for h in range(self.num_heads):
            k = kernel[:, h, :L].to(x.dtype)
            kh = km_cache[:, :L, h]
            vh = v_cache[:, :L, h, :]
            stacked = torch.cat([kh.unsqueeze(-1), kh.unsqueeze(-1) * vh], dim=-1)
            stacked = torch.flip(stacked, dims=[1])
            out_stacked = torch.einsum("bl,bld->bd", k, stacked)
            head_ctx = out_stacked[:, 1:] / (out_stacked[:, 0:1] + 1e-8)
            if content_ctx is not None:
                head_ctx = (1.0 - self.content_mix) * head_ctx + self.content_mix * (
                    content_ctx[:, h]
                )
            qh = qm_new[:, 0, h].unsqueeze(-1)
            v_last = v_new[:, 0, h, :]
            out_heads.append(qh * head_ctx + (1.0 - qh) * v_last)

        multihead_out = torch.cat(out_heads, dim=-1).unsqueeze(1)
        out = self.out_proj(multihead_out)
        if self.use_ffn:
            x = self.ln_attn(x + out)
            x = self.ln_ffn(x + self.ffn(x))
            mixed = x
        else:
            mixed = out + x

        cache = (x_cache, km_cache, v_cache, kernel, cache_len, content_S, content_Z)
        return mixed, cache

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, ...] | None = None,
        use_cache: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if use_cache:
            out, cache = self._forward_wda_cached(x, past_key_value)
            return out, cache
        B, S, D = x.shape
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        pooled = x.mean(dim=1)
        qm, km = torch.sigmoid(self.q_mod(x)), torch.sigmoid(self.k_mod(x))
        out_heads = []
        content_ctx = None

        # Reset routing stats
        collect_stats = bool(getattr(self, "collect_stats", True))
        all_head_active_masks = []
        all_head_entropies = []

        if self.content_mix > 0.0:
            if self.learned_content:
                q = (
                    self.content_q_proj(x)
                    .view(B, S, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                k = (
                    self.content_k_proj(x)
                    .view(B, S, self.num_heads, self.content_rank)
                    .transpose(1, 2)
                )
                content_ctx = self._causal_linear_attn_chunked(
                    self._phi(q), self._phi(k), v
                )
            else:
                xh = x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
                qf = self._phi(torch.einsum("bhsd,hdr->bhsr", xh, self.content_wq))
                kf = self._phi(torch.einsum("bhsd,hdr->bhsr", xh, self.content_wk))
                content_ctx = self._causal_linear_attn_chunked(qf, kf, v)

        for h in range(self.num_heads):
            logits = self.gatings[h](pooled)
            weights = topk_sparse_weights(
                logits, k=self.topk_masks, temperature=self.gate_temp
            )

            # --- Routing Diversity Metrics ---
            if collect_stats:
                with torch.no_grad():
                    # Count masks used at least once in the batch for this head
                    batch_active = (weights.sum(dim=0) > 0).float().sum().item()
                    all_head_active_masks.append(batch_active)
                    # Shannon entropy of routing probabilities (before top-k)
                    probs = torch.softmax(logits, dim=-1)
                    entropy = (
                        -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                    )
                    all_head_entropies.append(entropy)

            coeff = self.mod_mlp[h](pooled)
            scale = 0.5 * torch.tanh(
                torch.einsum("br,rm->bm", coeff, self.mod_basis[h])
            )
            weights_mod = (weights * (1.0 + scale)).clamp_min(0.0)
            weights_mod = weights_mod / (weights_mod.sum(dim=-1, keepdim=True) + 1e-8)
            weights_mod = weights_mod.to(dtype=x.dtype)
            vh = v[:, h]
            if self.use_checkpoint:
                attn_kern_batch = checkpoint(
                    lambda wm: self._render_attn_kernel(h, wm),
                    weights_mod,
                    use_reentrant=False,
                )
            else:
                attn_kern_batch = self._render_attn_kernel(h, weights_mod)

            # Causal part of the kernel starts at the middle (distance 0)
            mid = self.seq_len - 1
            attn_kern_pos = attn_kern_batch[:, mid : mid + S]

            if self.use_sampling:
                noise = (
                    torch.randn(B, self.num_samples, S, device=x.device, dtype=x.dtype)
                    * self.noise_sigma
                )
                dens_kern_pos = self.step(attn_kern_pos.unsqueeze(1) + noise).mean(
                    dim=1
                )
            else:
                dens_kern_pos = torch.sigmoid(self.attn_alpha * attn_kern_pos)

            dens_kern_pos = dens_kern_pos * self.window_kern_pos[:S].unsqueeze(0).to(
                dens_kern_pos.dtype
            )
            kh = km[:, :, h].unsqueeze(-1)
            stacked = torch.cat([kh, kh * vh], dim=-1)
            out_stacked = self._toeplitz_causal_apply_fft_stacked(
                dens_kern_pos, stacked
            )
            head_ctx = out_stacked[:, :, 1:] / (out_stacked[:, :, 0:1] + 1e-8)
            if content_ctx is not None:
                head_ctx = (
                    1.0 - self.content_mix
                ) * head_ctx + self.content_mix * content_ctx[:, h]
            qh = qm[:, :, h].unsqueeze(-1)
            out_heads.append(qh * head_ctx + (1.0 - qh) * vh)

        # Store mean stats
        if collect_stats and all_head_active_masks:
            self.routing_stats = {
                "active_masks": sum(all_head_active_masks) / len(all_head_active_masks),
                "entropy": sum(all_head_entropies) / len(all_head_entropies),
            }
        else:
            self.routing_stats = {}

        multihead_out = torch.cat(out_heads, dim=-1)
        out = self.out_proj(multihead_out)
        if self.use_ffn:
            x = self.ln_attn(x + out)
            x = self.ln_ffn(x + self.ffn(x))
            return x
        return out + x


class WaveCharLM(nn.Module):
    """Character-level causal LM using WaveDensityAttentionBlock."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 64,
        embed_dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        num_masks: int = 160,
        num_waves_per_mask: int = 40,
        topk_masks: int = 8,
        attn_alpha: float = 3.0,
        content_rank: int = 8,
        content_mix: float = 0.0,
        learned_content: bool = False,
        use_sin_waves: bool = True,
        ffn_mult: int = 4,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size, self.seq_len, self.embed_dim = (
            int(vocab_size),
            int(seq_len),
            int(embed_dim),
        )
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
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        x = self.tok_emb(idx) + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))
