import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "transplant_smol135m.json"
    ckpt_path = (
        root / "private" / "checkpoints" / "smol135m_wda" / "adapter_final_ema.pt"
    )
    out_dir = root / "export" / "smol135m-wda-ema"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    payload = torch.load(ckpt_path, map_location="cpu")
    adapter = (
        payload["adapter"]
        if isinstance(payload, dict) and "adapter" in payload
        else payload
    )

    # Save adapter weights as safetensors for easy distribution/loading.
    try:
        from safetensors.torch import save_file
    except Exception as e:
        raise RuntimeError(
            "safetensors is required for export. Install it (pip install safetensors) and retry."
        ) from e

    st_path = out_dir / "adapter_model.safetensors"
    save_file(adapter, str(st_path))

    # Minimal config for loader scripts
    export_cfg = {
        "base_model_name_or_path": cfg.get("teacher_model_id"),
        "seq_len": cfg.get("seq_len"),
        "wda_num_masks": cfg.get("wda_num_masks"),
        "wda_num_waves_per_mask": cfg.get("wda_num_waves_per_mask"),
        "wda_topk_masks": cfg.get("wda_topk_masks"),
        "wda_attn_alpha": cfg.get("wda_attn_alpha"),
        "wda_gate_temp": cfg.get("wda_gate_temp"),
        "wda_content_mix": cfg.get("wda_content_mix", 0.15),
        "wda_learned_content": cfg.get("wda_learned_content", True),
        "wda_use_sin_waves": cfg.get("wda_use_sin_waves", True),
        "wda_noise_sigma": cfg.get("wda_noise_sigma", 0.12),
        "wda_step_alpha": cfg.get("wda_step_alpha", 6.0),
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(export_cfg, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Convenience: copy a readme-ish note
    (out_dir / "README.md").write_text(
        "# SmolLM-135M WDA (EMA)\n\n"
        "This export contains WDA+gamma+backbone weights saved from `adapter_final_ema.pt`.\n\n"
        "Files:\n"
        "- `adapter_model.safetensors`: weights\n"
        "- `adapter_config.json`: WDA hyperparams + base model id\n\n"
        "To load, use `scripts/run_smol135m_wda_ema_inference.py` or adapt `export/.../load_wda.py`.\n",
        encoding="utf-8",
    )

    print(f"Exported: {st_path}")
    print(f"Wrote config: {out_dir / 'adapter_config.json'}")


if __name__ == "__main__":
    main()
