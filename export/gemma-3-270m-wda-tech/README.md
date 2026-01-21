# WDA Gemma-3-270M Specialized Tech (Step 1550)

This is a [Wave Density Attention (WDA)](https://github.com/conrad-v/WDA) transplant model based on `google/gemma-3-270m-it`.

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
model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 2. Setup WDA (Infrastructure code provided in this repo)
# Use patch_attention_modules followed by load_adapter with this path
```
