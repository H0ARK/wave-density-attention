import torch
import torch.nn as nn
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from transformers import AutoModelForCausalLM


def check_checkpoint(model_id, adapter_path):
    print(f"Checking checkpoint: {adapter_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
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
    try:
        payload = torch.load(adapter_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    state_dict = (
        payload["adapter"]
        if isinstance(payload, dict) and "adapter" in payload
        else payload
    )

    # Load parameters manually if needed
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"Shape mismatch for {k}: {model_state[k].shape} vs {v.shape}")
        else:
            # Maybe the key name changed?
            print(f"Missing key in model: {k}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded: Missing={len(missing)}, Unexpected={len(unexpected)}")

    wrapper_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, ParallelAttentionWrapper):
            # Check weight RMS
            wda_qkv_rms = m.wda_qkv.weight.data.pow(2).mean().sqrt().item()
            teacher_qkv_rms = (
                m.teacher_attn.q_proj.weight.data.pow(2).mean().sqrt().item()
            )
            gamma = m.gamma.item()
            eff_wda = gamma * wda_qkv_rms

            print(
                f"L{wrapper_idx:2d} | G:{gamma:.3f} | WDA_W:{wda_qkv_rms:.4f} | T_W:{teacher_qkv_rms:.4f} | EffW:{eff_wda:.4f}"
            )
            wrapper_idx += 1


if __name__ == "__main__":
    check_checkpoint(
        "Qwen/Qwen2.5-0.5B",
        "private/checkpoints/qwen05b_hero_sequential/adapter_step1400.pt",
    )
