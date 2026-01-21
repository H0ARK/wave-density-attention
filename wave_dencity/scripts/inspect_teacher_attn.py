import argparse
from typing import Iterable

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from wave_dencity.transplant.patching import find_attention_modules


def _iter_linear_weights(mod: nn.Module) -> Iterable[tuple[str, torch.Size]]:
    for name, child in mod.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child.weight.shape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dapn/gemma-3-270M-it-coder")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[args.dtype.lower()]

    print(f"Loading {args.model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="cpu",
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="cpu",
        )

    refs = find_attention_modules(model)
    print(f"Found {len(refs)} attention-like modules")

    for ref in refs:
        mod = ref.module
        print("-")
        print(f"{ref.path}: {mod.__class__.__name__}")
        for ln, shape in _iter_linear_weights(mod):
            # Keep output short; print the common projections if present.
            if any(k in ln for k in ["q", "k", "v", "o", "out", "proj", "qkv"]):
                print(f"  {ln}.weight: {tuple(shape)}")


if __name__ == "__main__":
    main()
