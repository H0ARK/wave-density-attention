import torch
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from transformers import AutoModelForCausalLM
import re


def check_checkpoint(model_id, adapter_path):
    print(f"Checking checkpoint: {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Wrap
    for name, module in model.named_modules():
        if "attn" in name and "." not in name and hasattr(module, "q_proj"):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            short_name = name.split(".")[-1]

            wrapped = ParallelAttentionWrapper(module)
            setattr(parent, short_name, wrapped)

    # Load adapter
    state_dict = torch.load(adapter_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Measure WDA weights vs Teacher
    for i, m in enumerate(model.modules()):
        if isinstance(m, ParallelAttentionWrapper):
            wda_qkv_rms = m.wda_qkv.weight.data.pow(2).mean().sqrt().item()
            teacher_qkv_rms = (
                m.teacher_attn.q_proj.weight.data.pow(2).mean().sqrt().item()
            )
            gamma = m.gamma.item()

            # Effective WDA RMS = gamma * WDA_Weight_RMS
            # (Simplified as it's linear)
            effective_wda = gamma * wda_qkv_rms

            print(
                f"Layer {i:2d}: Gamma={gamma:.3f} | WDA_W_RMS={wda_qkv_rms:.4f} | Teacher_W_RMS={teacher_qkv_rms:.4f} | Eff_WDA={effective_wda:.4f}"
            )


if __name__ == "__main__":
    check_checkpoint(
        "Qwen/Qwen2.5-0.5B",
        "private/checkpoints/qwen05b_hero_sequential/adapter_step1400.pt",
    )
