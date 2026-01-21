"""Attention-transplant utilities (HF teacher → WDA student)."""

from .config import TransplantConfig
from .patching import find_attention_modules, patch_attention_modules
from .parallel_attn import ParallelAttentionWrapper
from .wda import WDABridge
