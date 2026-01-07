---
language: en
license: apache-2.0
tags:
- wave-density
- causal-lm
- interference-attention
---

# Wave-Density Attention (WDA) — 130M Parameter Language Model

This repository contains a 130M parameter causal language model built with Wave-Density Attention (WDA), a novel alternative to standard dot-product self-attention.

WDA reframes attention as a wave-interference and density-rendering process, replacing the traditional $QK^\top$ similarity computation with learned frequency-based interactions. This allows attention patterns to emerge from constructive and destructive interference rather than explicit pairwise dot products.

⸻

## Model Overview
- **Architecture**: Decoder-only Transformer with Wave-Density Attention
- **Parameters**: ~130M
- **Context Length**: 256 tokens
- **Attention Mechanism**: Wave-Density Attention (Mixture-of-Masks via learned wave bases)
- **Training Regime**: From scratch

## Training Data
- **Primary**: UltraChat 200k (instruction-style supervision)
- **Initialization / Mixing**: Streaming C4 (broad web text)

This combination provides both general language coverage and instruction-following coherence, while allowing the WDA mechanism to learn stable long-range structure.

⸻

## Performance
- **Validation Loss (UltraChat)**: ~2.86
- **Equivalent Perplexity**: ~17.5–20 (best checkpoints)
- **Model Size**: 130M parameters

Despite using a fundamentally different attention formulation, WDA achieves competitive perplexity and strong qualitative coherence at this scale.

⸻

## Usage

To use this model, install or clone the reference implementation from the official repository:

👉 [**Wave-Density Attention code**](https://github.com/H0ARK/wave-density-attention)

Example loading snippet:

```python
from wave_dencity import WaveCharLM
import torch
import json

# Load model configuration
with open("config.json", "r") as f:
    config = json.load(f)

model = WaveCharLM(**config)
# Load weights from model.safetensors
# model.load_state_dict(...)
model.eval()
```

Note: This model is intended for research and experimentation with alternative attention mechanisms. The codebase exposes WDA internals for inspection and modification.

⸻

## Why Wave-Density Attention?

Traditional attention relies on sharp token-to-token similarity. WDA instead:
- Uses frequencies as a representational tool
- Produces attention surfaces via interference patterns
- Selects among multiple learned attention masks dynamically (Mixture-of-Masks / “MoM”)

This approach avoids explicit dot-product similarity while still supporting coherent, causal language modeling.

⸻

## Citation

If you use this model or the Wave-Density Attention mechanism in your work, please cite the official repository and paper.
