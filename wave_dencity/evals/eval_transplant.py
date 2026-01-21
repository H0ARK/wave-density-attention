import argparse
import math
import random
import re
import os
import sys

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import inspect
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wave_dencity.transplant.adapter import load_adapter
from wave_dencity.transplant.config import load_config, parse_torch_dtype
from wave_dencity.transplant.data import get_data_streamer
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.wda import WDABridge


def _load_causal_lm(model_id: str, *, dtype: torch.dtype, device: str):
    kwargs = {"torch_dtype": dtype}
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **kwargs
        ).to(device)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)


def _get_num_heads(cfg_obj) -> int:
    for key in ["num_attention_heads", "num_heads", "n_head"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_heads from model.config")


def _set_alpha(model: torch.nn.Module, alpha: float) -> None:
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.set_alpha(alpha)


def _set_scales(
    model: torch.nn.Module, *, teacher_scale: float, wda_scale: float
) -> None:
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.set_scales(teacher_scale=teacher_scale, wda_scale=wda_scale)


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


def _compile_wda_blocks(
    wrappers: list[ParallelAttentionWrapper],
    backend: str,
    mode: str,
    *,
    dynamic: bool = False,
    capture_scalar_outputs: bool = False,
    cache_size_limit: int | None = None,
) -> None:
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; skipping WDA compilation.")
        return
    if capture_scalar_outputs or cache_size_limit:
        try:
            import torch._dynamo as dynamo

            if capture_scalar_outputs:
                dynamo.config.capture_scalar_outputs = True
            if cache_size_limit is not None and cache_size_limit > 0:
                dynamo.config.cache_size_limit = max(
                    dynamo.config.cache_size_limit, int(cache_size_limit)
                )
        except Exception as exc:
            print(f"WARNING: Failed to set dynamo config: {exc}")
    if backend == "inductor":
        try:
            import triton  # noqa: F401
        except Exception:
            print(
                "Triton not available; skipping WDA compilation. "
                "Install triton or use --compile_backend aot_eager."
            )
            return
    compile_kwargs = {"backend": backend, "mode": mode}
    try:
        sig = inspect.signature(torch.compile)
        if dynamic and "dynamic" in sig.parameters:
            compile_kwargs["dynamic"] = True
    except Exception:
        pass

    compiled = 0
    for m in wrappers:
        wda = getattr(m, "wda_attn", None)
        block = getattr(wda, "block", None) if wda is not None else None
        if block is None:
            continue
        try:
            wda.block = torch.compile(block, **compile_kwargs)
            compiled += 1
        except Exception as exc:
            print(f"WARNING: WDA compile failed: {exc}")
            break
    if compiled:
        print(f"Compiled {compiled} WDA blocks (backend={backend}, mode={mode}).")


def _ppl_from_losses(losses: list[float]) -> float:
    m = sum(losses) / max(len(losses), 1)
    return math.exp(min(20.0, m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transplant_gemma270m.json")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--mode", choices=["teacher", "parallel", "wda"], default="parallel"
    )
    ap.add_argument("--teacher_scale", type=float, default=None)
    ap.add_argument("--wda_scale", type=float, default=None)
    ap.add_argument(
        "--patch_layers",
        default="all",
        help="Which decoder layer indices to patch (e.g. 'all' or '0' or '0,1,2')",
    )
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument(
        "--dataset_mode",
        choices=["auto", "path", "datasets"],
        default="auto",
        help="Select dataset source: auto uses cfg.datasets if present, else cfg.dataset_path.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for dataset mixing randomness (only affects cfg.datasets mixtures).",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--compile_wda",
        action="store_true",
        help="Compile WDA blocks with torch.compile for faster eval.",
    )
    ap.add_argument("--compile_backend", default="inductor")
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument(
        "--compile_dynamic",
        action="store_true",
        help="Allow dynamic shapes during compilation.",
    )
    ap.add_argument(
        "--dynamo_capture_scalar",
        action="store_true",
        help="Enable Dynamo capture of scalar outputs to reduce graph breaks.",
    )
    ap.add_argument(
        "--dynamo_cache_size",
        type=int,
        default=0,
        help="Increase Dynamo cache_size_limit (0 keeps default).",
    )
    ap.add_argument(
        "--disable_routing_stats",
        action="store_true",
        help="Disable WDA routing stats to reduce graph breaks.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device

    if args.seed is not None:
        random.seed(int(args.seed))

    device = str(cfg.device)
    dtype = parse_torch_dtype(cfg.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = _load_causal_lm(cfg.teacher_model_id, dtype=dtype, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = _load_causal_lm(cfg.teacher_model_id, dtype=dtype, device=device)
    student.eval()

    hidden_size = int(
        getattr(student.config, "hidden_size", getattr(student.config, "n_embd", None))
    )
    num_heads = _get_num_heads(student.config)

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
        return ParallelAttentionWrapper(attn_mod, wda, init_alpha=0.0)

    def filter_paths(path: str) -> bool:
        if patch_layer_set is None:
            return True
        idx = _infer_layer_index(path)
        return idx is not None and idx in patch_layer_set

    patched = patch_attention_modules(student, make_wrapper, filter_paths=filter_paths)
    print(f"Patched {len(patched)} attention modules")

    wrappers = [m for m in student.modules() if isinstance(m, ParallelAttentionWrapper)]

    for name, p in student.named_parameters():
        if ".wda_attn." in name or name.endswith(".gamma"):
            p.requires_grad = True

    if args.adapter is not None:
        info = load_adapter(student, args.adapter, strict=False)
        print(
            f"Loaded adapter: loaded={info.get('loaded', 0)} skipped={info.get('skipped', 0)} shape_mismatch={info.get('shape_mismatch', 0)}"
        )

    if args.compile_wda:
        _compile_wda_blocks(
            wrappers,
            args.compile_backend,
            args.compile_mode,
            dynamic=args.compile_dynamic,
            capture_scalar_outputs=args.dynamo_capture_scalar,
            cache_size_limit=(
                int(args.dynamo_cache_size) if args.dynamo_cache_size else None
            ),
        )

    if args.disable_routing_stats:
        for m in wrappers:
            block = getattr(getattr(m, "wda_attn", None), "block", None)
            if block is not None:
                block.collect_stats = False

    _set_alpha(student, float(args.alpha))

    if args.mode == "teacher":
        ts, ws = 1.0, 0.0
    elif args.mode == "wda":
        ts, ws = 0.0, 1.0
    else:
        ts, ws = 1.0, 1.0

    if args.teacher_scale is not None:
        ts = float(args.teacher_scale)
    if args.wda_scale is not None:
        ws = float(args.wda_scale)

    _set_scales(student, teacher_scale=ts, wda_scale=ws)
    print(f"mode={args.mode} alpha={args.alpha} teacher_scale={ts} wda_scale={ws}")

    dataset_cfg = cfg.dataset_path
    if args.dataset_mode == "datasets":
        if cfg.datasets is None:
            raise ValueError(
                "cfg.datasets is not set but dataset_mode='datasets' was requested"
            )
        dataset_cfg = cfg.datasets
    elif args.dataset_mode == "auto" and cfg.datasets is not None:
        dataset_cfg = cfg.datasets

    stream = get_data_streamer(
        tokenizer,
        dataset_cfg=dataset_cfg,
        seq_len=cfg.seq_len,
        micro_batch_size=cfg.micro_batch_size,
        device=device,
    )

    teacher_losses: list[float] = []
    student_losses: list[float] = []

    with torch.no_grad():
        for _ in range(args.batches):
            batch = next(stream)

            t_logits = teacher(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits
            s_logits = student(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits

            targets = batch.input_ids[:, 1:]
            mask = batch.attention_mask[:, 1:].to(torch.float32)

            t_ce = F.cross_entropy(
                t_logits[:, :-1, :].contiguous().view(-1, t_logits.size(-1)),
                targets.contiguous().view(-1),
                reduction="none",
            ).view_as(targets)
            s_ce = F.cross_entropy(
                s_logits[:, :-1, :].contiguous().view(-1, s_logits.size(-1)),
                targets.contiguous().view(-1),
                reduction="none",
            ).view_as(targets)

            t_loss = float((t_ce * mask).sum().item() / (mask.sum().item() + 1e-8))
            s_loss = float((s_ce * mask).sum().item() / (mask.sum().item() + 1e-8))

            teacher_losses.append(t_loss)
            student_losses.append(s_loss)

    print(
        f"Teacher CE: {sum(teacher_losses)/len(teacher_losses):.4f} | ppl={_ppl_from_losses(teacher_losses):.2f}"
    )
    print(
        f"Student CE: {sum(student_losses)/len(student_losses):.4f} | ppl={_ppl_from_losses(student_losses):.2f} (alpha={args.alpha})"
    )


if __name__ == "__main__":
    main()
