import argparse
import math
import re
from dataclasses import dataclass

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wave_dencity.transplant.adapter import load_adapter
from wave_dencity.transplant.config import load_config, parse_torch_dtype
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.wda import WDABridge


def _get_num_heads(cfg_obj) -> int:
    for key in ["num_attention_heads", "num_heads", "n_head"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_heads from model.config")


def _get_num_layers(cfg_obj) -> int:
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_layers from model.config")


def _load_causal_lm(model_id: str, *, dtype: torch.dtype, device: str):
    kwargs = {"torch_dtype": dtype}
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", **kwargs).to(device)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)


def _set_mode(model: torch.nn.Module, *, alpha: float, mode: str, teacher_scale: float | None, wda_scale: float | None) -> None:
    if mode == "teacher":
        ts, ws = 1.0, 0.0
    elif mode == "wda":
        ts, ws = 0.0, 1.0
    elif mode == "parallel":
        ts, ws = 1.0, 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if teacher_scale is not None:
        ts = float(teacher_scale)
    if wda_scale is not None:
        ws = float(wda_scale)

    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.set_alpha(float(alpha))
            m.set_scales(teacher_scale=ts, wda_scale=ws)


def _gamma_stats(model: torch.nn.Module) -> tuple[float, float, float]:
    vals = []
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            vals.append(float(m.gamma.detach().float().cpu().item()))
    if not vals:
        return 0.0, 0.0, 0.0
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / len(vals)
    return mean, mn, mx


_LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]


def _infer_layer_index(path: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        m = pat.search(path)
        if m:
            return int(m.group(1))
    return None


def _layer_alpha_scale(layer_idx: int, num_layers: int, mode: str) -> float:
    if num_layers <= 1:
        return 1.0
    t = layer_idx / float(num_layers - 1)
    if mode == "uniform":
        return 1.0
    if mode == "linear":
        return float(t)
    if mode == "sqrt":
        return float(math.sqrt(max(t, 0.0)))
    raise ValueError(f"Unknown alpha_layer_scale: {mode}")


def _make_window(token_ids: list[int], *, seq_len: int, pad_id: int, device: str):
    if len(token_ids) >= seq_len:
        window = token_ids[-seq_len:]
        attn = [1] * seq_len
    else:
        pad_n = seq_len - len(token_ids)
        window = [pad_id] * pad_n + token_ids
        attn = [0] * pad_n + [1] * len(token_ids)
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
    return input_ids, attention_mask


@torch.inference_mode()
def generate_sliding_window(
    model,
    tokenizer,
    *,
    prompt: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: str,
):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    context_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for _ in range(max_new_tokens):
        input_ids, attention_mask = _make_window(context_ids, seq_len=seq_len, pad_id=pad_id, device=device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        next_logits = logits[:, -1, :]

        if temperature and temperature > 0:
            next_logits = next_logits / float(temperature)

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            remove = cumprobs > float(top_p)
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            next_logits = torch.full_like(next_logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

        if do_sample:
            probs = F.softmax(next_logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
        else:
            nxt = int(torch.argmax(next_logits, dim=-1).item())

        context_ids.append(nxt)

    return context_ids


@dataclass
class Score:
    nll: float
    ppl: float
    avg_kl_vs_teacher: float


@torch.inference_mode()
def score_continuation(
    model,
    *,
    prompt_ids: list[int],
    continuation_ids: list[int],
    teacher_logits_fn,
    seq_len: int,
    pad_id: int,
    device: str,
):
    nlls: list[float] = []
    kls: list[float] = []

    context: list[int] = list(prompt_ids)

    for tok in continuation_ids:
        # Predict tok given current context.
        input_ids, attention_mask = _make_window(context, seq_len=seq_len, pad_id=pad_id, device=device)
        s_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
        t_logits = teacher_logits_fn(context)

        logp = F.log_softmax(s_logits, dim=-1)[0, tok]
        nlls.append(float(-logp.item()))

        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        kls.append(float(kl.item()))

        context.append(tok)

    nll = float(sum(nlls) / max(len(nlls), 1))
    ppl = float(math.exp(min(20.0, nll)))
    avg_kl = float(sum(kls) / max(len(kls), 1))
    return Score(nll=nll, ppl=ppl, avg_kl_vs_teacher=avg_kl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transplant_gemma270m.json")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default="Write a Python function to compute Fibonacci numbers.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mode", choices=["teacher", "parallel", "wda", "all"], default="all")
    ap.add_argument("--teacher_scale", type=float, default=None)
    ap.add_argument("--wda_scale", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0=greedy")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gamma_override", type=float, default=None, help="Force all wrapper gammas to this value (debug)")
    ap.add_argument(
        "--patch_layers",
        default="all",
        help="Which decoder layer indices to patch (e.g. 'all' or '0' or '0,1,2')",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device

    device = str(cfg.device)
    dtype = parse_torch_dtype(cfg.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_causal_lm(cfg.teacher_model_id, dtype=dtype, device=device)
    model.eval()

    # 1. Generate reference text using UNPATCHED teacher (official generate)
    print(f"\n--- Generating reference continuation with unpatched teacher ---")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    with torch.no_grad():
        ref_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    ref_ids_list = ref_out[0].tolist()
    prompt_len = prompt_ids.shape[1]
    continuation_ids = ref_ids_list[prompt_len:]
    
    print(f"Prompt tokens: {prompt_len} | Continuation tokens: {len(continuation_ids)}")
    print(f"Teacher Baseline Text: {tokenizer.decode(ref_ids_list, skip_special_tokens=True)}")

    # 2. Extract unpatched logits for the continuation (for KL scoring)
    # We do this before patching to have a perfect reference
    @torch.inference_mode()
    def get_teacher_logits(ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], device=device)
        return model(input_ids).logits[:, -1, :]

    # 3. Patch model for WDA/Parallel testing
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None)))
    num_heads = _get_num_heads(model.config)
    num_layers = _get_num_layers(model.config)
    
    if str(args.patch_layers).lower() == "all":
        patch_layer_set: set[int] | None = None
    else:
        patch_layer_set = set()
        for part in str(args.patch_layers).split(","):
            part = part.strip()
            if not part:
                continue
            patch_layer_set.add(int(part))

    def make_wrapper(attn_mod: torch.nn.Module, path: str) -> torch.nn.Module:
        layer_idx = _infer_layer_index(path)
        if layer_idx is None:
            layer_scale = 1.0
        else:
            layer_scale = _layer_alpha_scale(layer_idx, num_layers, cfg.alpha_layer_scale)

        wda = WDABridge(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=cfg.seq_len,
            num_masks=cfg.wda_num_masks,
            num_waves_per_mask=cfg.wda_num_waves_per_mask,
            topk_masks=cfg.wda_topk_masks,
            attn_alpha=cfg.wda_attn_alpha,
            content_mix=cfg.wda_content_mix,
            learned_content=cfg.wda_learned_content,
            use_sin_waves=cfg.wda_use_sin_waves,
            use_sampling=cfg.wda_use_sampling,
            num_samples=cfg.wda_num_samples,
            noise_sigma=cfg.wda_noise_sigma,
            step_alpha=cfg.wda_step_alpha,
            use_checkpoint=cfg.wda_use_checkpoint,
        )
        return ParallelAttentionWrapper(attn_mod, wda, init_alpha=0.0, init_layer_alpha_scale=layer_scale)

    def filter_paths(path: str) -> bool:
        if patch_layer_set is None:
            return True
        idx = _infer_layer_index(path)
        return idx is not None and idx in patch_layer_set

    patched = patch_attention_modules(model, make_wrapper, filter_paths=filter_paths)
    print(f"Patched {len(patched)} attention modules")

    # Unfreeze WDA+gamma so adapter loads.
    for name, p in model.named_parameters():
        if ".wda_attn." in name or name.endswith(".gamma"):
            p.requires_grad = True

    if args.adapter is not None:
        info = load_adapter(model, args.adapter, strict=False)
        print(
            f"Loaded adapter: loaded={info.get('loaded', 0)} skipped={info.get('skipped', 0)} shape_mismatch={info.get('shape_mismatch', 0)}"
        )

    if args.gamma_override is not None:
        for m in model.modules():
            if isinstance(m, ParallelAttentionWrapper):
                m.gamma.data.fill_(float(args.gamma_override))

    model.eval()

    modes = [args.mode] if args.mode != "all" else ["teacher", "parallel", "wda"]

    print(f"seq_len={cfg.seq_len} alpha={args.alpha} adapter={args.adapter}")
    g_mean, g_min, g_max = _gamma_stats(model)
    print(f"gamma: mean={g_mean:.4f} min={g_min:.4f} max={g_max:.4f}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # Score + (optionally) generate qualitative text per mode.
    for mode in modes:
        _set_mode(
            model,
            alpha=float(args.alpha),
            mode=mode,
            teacher_scale=args.teacher_scale,
            wda_scale=args.wda_scale,
        )

        # Generate under this mode using high-level model.generate for stability
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=min(64, args.max_new_tokens),
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                pad_token_id=pad_id,
            )

        # Score the reference continuation under this mode.
        score = score_continuation(
            model,
            prompt_ids=ref_ids_list[:prompt_len],
            continuation_ids=continuation_ids,
            teacher_logits_fn=get_teacher_logits,
            seq_len=cfg.seq_len,
            pad_id=pad_id,
            device=device,
        )

        text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        print("\n===", mode, "===")
        print(f"NLL={score.nll:.4f} PPL={score.ppl:.2f} avg_KL_vs_teacher={score.avg_kl_vs_teacher:.6f}")
        print(text)



if __name__ == "__main__":
    main()
