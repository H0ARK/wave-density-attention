import torch
import torch.nn as nn
from pathlib import Path
import re


def stretch_adapter(input_path, output_path, new_num_masks=64):
    print(f"Loading adapter from {input_path}")
    payload = torch.load(input_path, map_location="cpu")
    if "adapter" not in payload:
        print("Error: 'adapter' key not found in payload.")
        return

    adapter = payload["adapter"]
    new_adapter = {}

    # We assume current num_masks is 32.
    # We will identify the dimension to stretch by looking at the current shapes.
    # Keys for mask expansion:
    # 1. block.gatings.H.2.weight: [num_masks, gate_hidden] -> dim 0
    # 2. block.gatings.H.2.bias: [num_masks] -> dim 0
    # 3. block.mod_basis.H: [mod_rank, num_masks] -> dim 1
    # 4. block.freqs.H: [num_masks, num_waves, 2] -> dim 0
    # 5. block.amps.H: [num_masks, num_waves] -> dim 0
    # 6. block.phases.H: [num_masks, num_waves] -> dim 0

    for k, v in adapter.items():
        original_v = v.clone()

        # Determine if and how to stretch
        if ".gatings." in k and ".2.weight" in k:
            # [32, 128] -> [64, 128]
            old_num = v.shape[0]
            repeats = (new_num_masks + old_num - 1) // old_num
            v = torch.cat([v] * repeats, dim=0)[:new_num_masks]
            print(f"Stretched {k}: {original_v.shape} -> {v.shape}")

        elif ".gatings." in k and ".2.bias" in k:
            # [32] -> [64]
            old_num = v.shape[0]
            repeats = (new_num_masks + old_num - 1) // old_num
            v = torch.cat([v] * repeats, dim=0)[:new_num_masks]
            print(f"Stretched {k}: {original_v.shape} -> {v.shape}")

        elif ".mod_basis." in k:
            # [8, 32] -> [8, 64]
            old_num = v.shape[1]
            repeats = (new_num_masks + old_num - 1) // old_num
            v = torch.cat([v] * repeats, dim=1)[:new_num_masks]
            print(f"Stretched {k}: {original_v.shape} -> {v.shape}")

        elif (
            k.endswith(".freqs.0")
            or k.endswith(".freqs.1")
            or k.endswith(".freqs.2")
            or k.endswith(".freqs.3")
            or k.endswith(".amps.0")
            or k.endswith(".amps.1")
            or k.endswith(".amps.2")
            or k.endswith(".amps.3")
            or k.endswith(".phases.0")
            or k.endswith(".phases.1")
            or k.endswith(".phases.2")
            or k.endswith(".phases.3")
        ):
            # These are ParameterList items. Key ends with .freqs.H etc.
            # Shape is [num_masks, num_waves, ...] -> dim 0
            old_num = v.shape[0]
            repeats = (new_num_masks + old_num - 1) // old_num
            v = torch.cat([v] * repeats, dim=0)[:new_num_masks]
            print(f"Stretched {k}: {original_v.shape} -> {v.shape}")

        new_adapter[k] = v

    payload["adapter"] = new_adapter
    # We can keep the step or reset it. User wants to continue from 1550.
    # Let's keep the step.
    torch.save(payload, output_path)
    print(f"Saved expanded adapter to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--new_masks", type=int, default=64)
    args = parser.parse_args()

    stretch_adapter(args.input, args.output, args.new_masks)
