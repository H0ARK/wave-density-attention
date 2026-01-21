import torch
import re
from wave_dencity.transplant.adapter import load_adapter
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from transformers import AutoModelForCausalLM


def check_gamma(path):
    print(f"Checking {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "adapter" in checkpoint:
        adapter = checkpoint["adapter"]
    else:
        adapter = checkpoint

    # Just look at the state dict keys for gamma
    gamma_entries = {k: v for k, v in adapter.items() if "gamma" in k}
    if gamma_entries:
        # Sort keys naturally (by layer index if possible)
        def get_layer_idx(name):
            match = re.search(r"layers?\.(\d+)\.", name)
            return int(match.group(1)) if match else name

        sorted_keys = sorted(gamma_entries.keys(), key=get_layer_idx)

        print(f"Layer-by-layer Gammas for {path}:")
        for k in sorted_keys:
            val = gamma_entries[k].item()
            print(f"  {k:50} : {val:.6f}")

        values = list(gamma_entries.values())
        gammas_t = torch.stack(values)
        avg = gammas_t.mean().item()
        min_g = gammas_t.min().item()
        max_g = gammas_t.max().item()
        print(f"\nSummary: Avg: {avg:.6f} | Min: {min_g:.6f} | Max: {max_g:.6f}\n")
    else:
        print("No gamma found in state dict.")


if __name__ == "__main__":
    import os

    base = "private/checkpoints/transplant_gemma270m_specialized_tech"
    check_gamma(os.path.join(base, "adapter_step1000.pt"))
    check_gamma(os.path.join(base, "adapter_step2025.pt"))
