from __future__ import annotations

from typing import Any

from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper


def collect_baseline_stats(
    wrappers: list[ParallelAttentionWrapper], *, include_rms: bool = False
) -> list[dict[str, Any]]:
    stats = []
    for m in wrappers:
        ts_val = float(m.teacher_scale.item())
        ws_val = float(m.wda_scale.item())
        a_val = float((m.alpha * m.layer_alpha_scale).item())
        g_val = float(m.gamma.item())
        wda_gain = float(m.wda_gain.item())
        s_scale = ws_val * a_val * g_val * wda_gain
        denom = ts_val**2 + s_scale**2
        baseline_pct = 0.0 if denom <= 0.0 else (ts_val**2) / denom
        if include_rms:
            attn_rms = getattr(m, "last_attn_rms", 0.0)
            wda_rms = getattr(m, "last_wda_rms", 0.0)
            try:
                attn_rms = float(attn_rms) if attn_rms is not None else 0.0
            except Exception:
                attn_rms = 0.0
            try:
                wda_rms = float(wda_rms) if wda_rms is not None else 0.0
            except Exception:
                wda_rms = 0.0
            rms_ratio = 0.0 if attn_rms <= 0.0 else (wda_rms / (attn_rms + 1e-8))
            eff_s = abs(s_scale) * rms_ratio
            eff_denom = ts_val**2 + eff_s**2
            eff_baseline_pct = 0.0 if eff_denom <= 0.0 else (ts_val**2) / eff_denom
        else:
            attn_rms = 0.0
            wda_rms = 0.0
            rms_ratio = 0.0
            eff_s = 0.0
            eff_baseline_pct = 0.0
        stats.append(
            {
                "layer_idx": m.layer_idx,
                "baseline_pct": baseline_pct,
                "gamma": g_val,
                "wda_gain": wda_gain,
                "s_scale": s_scale,
                "attn_rms": attn_rms,
                "wda_rms": wda_rms,
                "rms_ratio": rms_ratio,
                "eff_s": eff_s,
                "eff_baseline_pct": eff_baseline_pct,
            }
        )
    return stats


def summarize_baseline(
    stats: list[dict[str, Any]], *, topk: int = 5
) -> tuple[
    float, float, float, float, float, float, float, float, list[dict[str, Any]]
]:
    if not stats:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
    avg_baseline = sum(s["baseline_pct"] for s in stats) / len(stats)
    avg_abs_s = sum(abs(s["s_scale"]) for s in stats) / len(stats)
    avg_eff_baseline = sum(s["eff_baseline_pct"] for s in stats) / len(stats)
    avg_eff_s = sum(abs(s["eff_s"]) for s in stats) / len(stats)
    avg_rms_ratio = sum(s["rms_ratio"] for s in stats) / len(stats)
    avg_attn_rms = sum(s["attn_rms"] for s in stats) / len(stats)
    avg_wda_rms = sum(s["wda_rms"] for s in stats) / len(stats)
    avg_wda_gain = sum(s["wda_gain"] for s in stats) / len(stats)
    top = sorted(stats, key=lambda s: s["baseline_pct"], reverse=True)[: max(0, topk)]
    top_list = [
        {
            "layer": s["layer_idx"],
            "baseline_pct": s["baseline_pct"],
            "eff_baseline_pct": s["eff_baseline_pct"],
            "gamma": s["gamma"],
            "wda_gain": s["wda_gain"],
            "s_scale": s["s_scale"],
            "attn_rms": s["attn_rms"],
            "wda_rms": s["wda_rms"],
            "rms_ratio": s["rms_ratio"],
            "eff_s": s["eff_s"],
        }
        for s in top
    ]
    return (
        avg_baseline,
        avg_abs_s,
        avg_eff_baseline,
        avg_eff_s,
        avg_rms_ratio,
        avg_attn_rms,
        avg_wda_rms,
        avg_wda_gain,
        top_list,
    )
