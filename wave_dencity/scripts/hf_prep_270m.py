import torch
from safetensors.torch import save_file
import json
import os
from pathlib import Path
import shutil


def convert_to_hf_ready(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    base_model_id: str = "google/gemma-3-270m-it",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Adapter Weights
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle state_dict if it's wrapped in a 'state_dict' or 'model' key
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove non-tensor keys for safetensors compliance
    keys_to_remove = [
        k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)
    ]
    for k in keys_to_remove:
        print(f"Removing non-tensor key: {k}")
        state_dict.pop(k)

    # 2. Save as Safetensors
    # We want to make sure keys are clean. Our keys look like
    # 'model.layers.0.self_attn.wda_attn.block.weight' etc.
    save_file(state_dict, output_path / "adapter_model.safetensors")
    print(f"Saved weights to {output_path / 'adapter_model.safetensors'}")

    # 3. Load and Filter Config
    with open(config_path, "r") as f:
        full_cfg = json.load(f)

    # Extract only the architecture relevant parts
    wda_config = {
        "base_model_name_or_path": base_model_id,
        "wda_num_masks": full_cfg.get("wda_num_masks", 64),
        "wda_num_waves_per_mask": full_cfg.get("wda_num_waves_per_mask", 4),
        "wda_topk_masks": full_cfg.get("wda_topk_masks", 8),
        "wda_attn_alpha": full_cfg.get("wda_attn_alpha", 3.0),
        "wda_content_mix": full_cfg.get("wda_content_mix", 0.1),
        "wda_learned_content": full_cfg.get("wda_learned_content", True),
        "wda_use_sin_waves": full_cfg.get("wda_use_sin_waves", True),
        "wda_noise_sigma": full_cfg.get("wda_noise_sigma", 0.12),
        "wda_step_alpha": full_cfg.get("wda_step_alpha", 6.0),
        "alpha_layer_scale": full_cfg.get("alpha_layer_scale", "sqrt"),
        "seq_len": full_cfg.get("seq_len", 512),
        "peft_type": "WDA_TRANSPLANT",  # Custom identifier
    }

    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(wda_config, f, indent=4)
    print(f"Saved config to {output_path / 'adapter_config.json'}")

    # 4. Copy current implementation files for "trust_remote_code" portability
    # Create a subfolder for the implementation
    src_dir = Path("wave_dencity")
    dst_dir = output_path / "wave_dencity"
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # Create the README
    readme_content = f"""# WDA Gemma-3-270M Specialized Tech (Step 1550)

This is a [Wave Density Attention (WDA)](https://github.com/conrad-v/wave-dence-atantion) transplant model based on `google/gemma-3-270m-it`.

## Model Description
This model uses a 64-mask WDA architecture substituted into the attention layers to provide dense, lithographical context resolution. It was trained on a mixture of:
- Specialized Technical reasoning data (IFEval instructions)
- STEM-focused pretraining data (InfiniByte Reasoning, Scientific Coding, Math Textbooks)

Step 1550 represents the state before the logic collapse observed later in training, retaining the most balanced trade-off between technical specialization and base model reasoning.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wave_dencity.transplant.adapter import load_adapter # Requires library
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper

# 1. Load Base
model_id = "{base_model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Setup WDA (Infrastructure code provided in this repo)
# Use patch_attention_modules followed by load_adapter with this path
```
"""
    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    convert_to_hf_ready(
        checkpoint_path="private/checkpoints/transplant_gemma270m_specialized_tech/adapter_step1550.pt",
        config_path="private/checkpoints/transplant_gemma270m_specialized_tech/config_used.json",
        output_dir="export/gemma-3-270m-wda-tech",
    )
