import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM

from wave_dencity.transplant.adapter import load_adapter, save_adapter_state
from wave_dencity.transplant.config import load_config, parse_torch_dtype
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


def _default_out_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("private/eval") / f"baseline_pct_{stamp}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transplant_gemma270m.json")
    ap.add_argument("--adapter", default=None, help="Path to adapter_step*.pt")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--mode", choices=["teacher", "parallel", "wda"], default="parallel"
    )
    ap.add_argument("--teacher_scale", type=float, default=None)
    ap.add_argument("--wda_scale", type=float, default=None)
    ap.add_argument(
        "--gamma_floor",
        type=float,
        default=None,
        help="Optional sign-preserving |gamma| floor to apply before computing baseline_pct.",
    )
    ap.add_argument(
        "--gamma_target_pct",
        type=float,
        default=0.2,
        help="Target baseline_pct for computing required |gamma| (default: 0.2).",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--export_adapter",
        default=None,
        help="Optional output path to save a gamma-floored adapter state.",
    )
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device

    device = str(cfg.device)
    dtype = parse_torch_dtype(cfg.torch_dtype)

    model = _load_causal_lm(cfg.teacher_model_id, dtype=dtype, device=device)

    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    )
    num_heads = _get_num_heads(model.config)

    def make_wrapper(attn_mod: torch.nn.Module, path: str) -> torch.nn.Module:
        layer_idx = _infer_layer_index(path)
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
        wrapper = ParallelAttentionWrapper(
            attn_mod, wda, init_alpha=0.0, layer_idx=layer_idx
        )
        wrapper.path = path
        return wrapper

    patched = patch_attention_modules(model, make_wrapper)
    print(f"Patched {len(patched)} attention modules")

    for name, p in model.named_parameters():
        if ".wda_attn." in name or name.endswith(".gamma"):
            p.requires_grad = True

    if args.adapter is not None:
        info = load_adapter(model, args.adapter, strict=False)
        print(
            "Loaded adapter: loaded={loaded} skipped={skipped} shape_mismatch={shape_mismatch}".format(
                **info
            )
        )

    _set_alpha(model, float(args.alpha))

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

    _set_scales(model, teacher_scale=ts, wda_scale=ws)

    gamma_floor = args.gamma_floor
    rows = []
    required_gammas: list[float] = []
    applied_gamma_clamps = 0
    for m in model.modules():
        if not isinstance(m, ParallelAttentionWrapper):
            continue
        layer_idx = m.layer_idx
        path = getattr(m, "path", None)
        ts_val = float(m.teacher_scale.item())
        ws_val = float(m.wda_scale.item())
        a_val = float((m.alpha * m.layer_alpha_scale).item())
        g_raw = float(m.gamma.item())
        g_val = g_raw
        if gamma_floor is not None:
            sign = -1.0 if g_raw < 0 else 1.0
            g_val = sign * max(abs(g_raw), float(gamma_floor))
            if g_val != g_raw:
                applied_gamma_clamps += 1
                m.gamma.data.fill_(g_val)

        s_scale = ws_val * a_val * g_val
        denom = ts_val**2 + s_scale**2
        baseline_pct = 0.0 if denom <= 0.0 else (ts_val**2) / denom

        req_gamma = None
        if ws_val != 0.0 and a_val != 0.0 and args.gamma_target_pct > 0.0:
            target = float(args.gamma_target_pct)
            if 0.0 < target < 1.0:
                req_s = ts_val * math.sqrt((1.0 - target) / target)
                req_gamma = req_s / (abs(ws_val * a_val) + 1e-12)
                required_gammas.append(req_gamma)

        rows.append(
            {
                "layer_idx": layer_idx,
                "path": path,
                "teacher_scale": ts_val,
                "wda_scale": ws_val,
                "alpha": a_val,
                "gamma_raw": g_raw,
                "gamma": g_val,
                "gamma_floor": gamma_floor,
                "required_gamma_abs_for_target": req_gamma,
                "student_scale": s_scale,
                "baseline_pct": baseline_pct,
            }
        )

    rows.sort(
        key=lambda r: (r["layer_idx"] is None, r["layer_idx"] or 0, r["path"] or "")
    )

    if rows:
        avg = sum(r["baseline_pct"] for r in rows) / len(rows)
        print(f"baseline_pct avg={avg:.6f} (n={len(rows)})")
        if required_gammas:
            req_sorted = sorted(required_gammas)
            mid = len(req_sorted) // 2
            median = (
                req_sorted[mid]
                if len(req_sorted) % 2 == 1
                else 0.5 * (req_sorted[mid - 1] + req_sorted[mid])
            )
            mean = sum(req_sorted) / len(req_sorted)
            print(
                "required |gamma| for target={:.3f}: median={:.6g} mean={:.6g} (n={})".format(
                    args.gamma_target_pct, median, mean, len(req_sorted)
                )
            )
        if gamma_floor is not None:
            print(f"gamma_floor applied to {applied_gamma_clamps} layers")
        for r in rows:
            lid = r["layer_idx"] if r["layer_idx"] is not None else "?"
            print(
                f"layer={lid:<3} baseline_pct={r['baseline_pct']:.6f} "
                f"ts={r['teacher_scale']:.6g} s_scale={r['student_scale']:.6g} "
                f"gamma={r['gamma']:.6g} path={r['path']}"
            )
    else:
        print("No ParallelAttentionWrapper modules found.")

    out_path = Path(args.out) if args.out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config),
        "adapter": args.adapter,
        "alpha": float(args.alpha),
        "mode": args.mode,
        "teacher_scale": ts,
        "wda_scale": ws,
        "gamma_floor": gamma_floor,
        "gamma_target_pct": args.gamma_target_pct,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.export_adapter is not None:
        adapter_state = {}
        for name, param in model.named_parameters():
            if ".wda_attn." in name or name.endswith(".gamma"):
                adapter_state[name] = param.detach().cpu()
        save_adapter_state(
            adapter_state,
            args.export_adapter,
            step=None,
            gamma_floor=gamma_floor,
            gamma_target_pct=args.gamma_target_pct,
        )
        print(f"Exported gamma-floored adapter to {args.export_adapter}")


if __name__ == "__main__":
    main()
