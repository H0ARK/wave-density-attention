import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

# Import the WDA library components (assumed to be in the same folder or installed)
from wave_dencity.transplant.adapter import (
    load_adapter,
    offload_teacher_attention,
    set_parallel_attn_mode,
)
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.wda import WDABridge


def load_wda_from_export(export_dir: str, device: str = "cuda"):
    exp_path = Path(export_dir)

    # 1. Load config
    with open(exp_path / "adapter_config.json", "r") as f:
        cfg = json.load(f)

    base_model_id = cfg["base_model_name_or_path"]
    print(f"Loading base model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if "gemma-3" in base_model_id.lower() else "sdpa",
    ).to(device)

    # 2. Patch with WDA architecture
    def make_wrapper(attn_mod, path):
        wda = WDABridge(
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            seq_len=cfg["seq_len"],
            num_masks=cfg["wda_num_masks"],
            num_waves_per_mask=cfg["wda_num_waves_per_mask"],
            topk_masks=cfg["wda_topk_masks"],
            attn_alpha=cfg["wda_attn_alpha"],
            content_mix=cfg["wda_content_mix"],
            learned_content=cfg["wda_learned_content"],
            use_sin_waves=cfg["wda_use_sin_waves"],
            noise_sigma=cfg["wda_noise_sigma"],
            step_alpha=cfg["wda_step_alpha"],
        )
        return ParallelAttentionWrapper(
            attn_mod, wda, init_alpha=1.0, init_teacher_scale=0.0, init_wda_scale=1.0
        )

    patch_attention_modules(model, make_wrapper)

    # Default to WDA-only cacheless mode (teacher output disabled).
    # If you want to use HF KV-cache for fast generation, keep teacher_scale>0
    # or implement KV-cache inside WDA.
    set_parallel_attn_mode(
        model,
        alpha=1.0,
        teacher_scale=0.0,
        wda_scale=1.0,
        gate_temp=cfg.get("wda_gate_temp"),
    )
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    # 3. Load weights
    adapter_path = exp_path / "adapter_model.safetensors"
    print(f"Loading WDA weights from {adapter_path}")
    from safetensors.torch import load_file

    state_dict = load_file(adapter_path)
    model.load_state_dict(state_dict, strict=False)

    # Offload the frozen teacher attention branch to CPU to reclaim VRAM.
    # Safe as long as you run with use_cache=False.
    offload_teacher_attention(model, device="cpu")

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_wda_from_export(".")
    prompt = "Explain the connection between WDA and lithographical attention."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(out[0]))
