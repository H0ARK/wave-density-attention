import torch
from train_transplant import (
    load_config,
    _load_causal_lm,
    _get_num_heads,
    _get_num_layers,
    patch_attention_modules,
    _get_num_heads,
    _infer_layer_index,
    _layer_alpha_scale,
)
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.wda import WDABridge


def debug_keys(config_path, adapter_path):
    cfg = load_config(config_path)
    model = _load_causal_lm(cfg.teacher_model_id, dtype=torch.bfloat16, device="cpu")

    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 4096))
    )
    num_heads = _get_num_heads(model.config)
    num_layers = _get_num_layers(model.config)

    def make_wrapper(attn_mod, path):
        layer_idx = _infer_layer_index(path)
        wda = WDABridge(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=cfg.seq_len,
            num_masks=cfg.wda_num_masks,
            num_waves_per_mask=cfg.wda_num_waves_per_mask,
            topk_masks=cfg.wda_topk_masks,
            attn_alpha=cfg.wda_attn_alpha,
        )
        return ParallelAttentionWrapper(attn_mod, wda, layer_idx=layer_idx)

    patch_attention_modules(model, make_wrapper)

    model_keys = set(model.state_dict().keys())

    payload = torch.load(adapter_path, map_location="cpu")
    adapter_keys = set(
        payload["adapter"].keys() if "adapter" in payload else payload.keys()
    )

    print(f"Model keys total: {len(model_keys)}")
    print(f"Adapter keys total: {len(adapter_keys)}")

    common = model_keys.intersection(adapter_keys)
    print(f"Common keys: {len(common)}")

    if len(common) == 0:
        print("\nSample Model keys:")
        for k in sorted(list(model_keys))[:10]:
            print(f"  {k}")
        print("\nSample Adapter keys:")
        for k in sorted(list(adapter_keys))[:10]:
            print(f"  {k}")


if __name__ == "__main__":
    debug_keys(
        "configs/qwen05b_hero_sequential.json",
        "private/checkpoints/qwen05b_hero_sequential/adapter_step1400.pt",
    )
