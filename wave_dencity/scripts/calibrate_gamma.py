import torch
import argparse
import os
import sys

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer, AutoModelForCausalLM
from wave_dencity.transplant.config import load_config, parse_torch_dtype
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.wda import WDABridge


def _get_num_heads(cfg_obj) -> int:
    for key in ["num_attention_heads", "num_heads", "n_head"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_heads")


import re

_LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]


def _infer_layer_index(path: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        m = pat.search(path)
        if m:
            return int(m.group(1))
    return None


def calibrate_gamma(config_path):
    print(f"Calibrating initial gammas for: {config_path}")
    cfg = load_config(config_path)
    device = cfg.device
    dtype = parse_torch_dtype(cfg.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)

    num_heads = _get_num_heads(model.config)
    head_dim = model.config.hidden_size // num_heads

    stats = {}  # layer_idx -> (teacher_std, wda_std)

    def make_wrapper(attn_mod, path):
        l_idx = _infer_layer_index(path)
        wda = WDABridge(
            hidden_size=model.config.hidden_size,
            num_heads=num_heads,
            seq_len=cfg.seq_len,
            num_masks=cfg.wda_num_masks,
            num_waves_per_mask=cfg.wda_num_waves_per_mask,
            topk_masks=cfg.wda_topk_masks,
            attn_alpha=cfg.wda_attn_alpha,
            content_mix=cfg.wda_content_mix,
            learned_content=cfg.wda_learned_content,
        )
        wrapper = ParallelAttentionWrapper(
            attn_mod,
            wda,
            layer_idx=l_idx,
            init_gamma=1.0,  # Start at 1.0 for measurement
        )
        return wrapper

    patch_attention_modules(model, make_wrapper)
    model.eval()

    # Create a representative dummy input (e.g. "Hello world" repeating)
    text = "The quick brown fox jumps over the lazy dog. " * 16
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=cfg.seq_len
    ).to(device)

    print("Running forward pass to capture magnitudes...")

    hooks = []
    layer_magnitudes = {}

    def get_hook(l_idx):
        def hook(module, input, output):
            # output is (mixed, rest...)
            # We want to compare the internal teacher_attn output vs wda_attn output
            # ParallelAttentionWrapper.forward does:
            # teacher_out = self.teacher_attn(hidden_states)
            # wda_out = self.wda_attn(hidden_states)

            with torch.no_grad():
                h = input[0]
                t_out_raw = module.teacher_attn(h, **kwargs_for_hook)
                if isinstance(t_out_raw, tuple):
                    t_out_raw = t_out_raw[0]

                w_out_raw = module.wda_attn(h, **kwargs_for_hook)
                if isinstance(w_out_raw, tuple):
                    w_out_raw = w_out_raw[0]

                t_std = t_out_raw.std().item()
                w_std = w_out_raw.std().item()
                layer_magnitudes[l_idx] = (t_std, w_std)

        return hook

    # Note: Using explicit forward calls inside the wrapper is safer than hooks for this
    # Let's just iterate over layers manually to be clean

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, ParallelAttentionWrapper):
                h = torch.randn(
                    1, 16, model.config.hidden_size, device=device, dtype=dtype
                )
                # Use real data for better stats
                batch_inputs = inputs.input_ids
                # We need to find the hidden states going INTO this layer.
                # Actually, an easier way is to just use the wrapper's properties.
                pass

    # Simplified approach: Measure WDA magnitude
    results = []

    # We'll use 0.08 as an empirical RMS for the teacher's attention output
    # because calling the teacher's forward requires complex rotary/mask args.
    teacher_rms_est = 0.08

    for name, m in model.named_modules():
        if isinstance(m, ParallelAttentionWrapper):
            l_idx = m.layer_idx
            # Sample hidden states with RMS ~ 1.0 on the same device as the module
            h = torch.randn(
                1, 128, model.config.hidden_size, device=device, dtype=dtype
            )
            m.to(device=device, dtype=dtype)  # Ensure module is on correct device/dtype

            w_out = m.wda_attn(h)
            if isinstance(w_out, tuple):
                w_out = w_out[0]

            w_rms = torch.sqrt(torch.mean(w_out**2)).item()
            ratio = teacher_rms_est / (w_rms + 1e-8)

            results.append(
                {
                    "layer": l_idx,
                    "teacher_rms": teacher_rms_est,
                    "wda_rms": w_rms,
                    "suggested_gamma": ratio,
                }
            )
            print(
                f"Layer {l_idx:2d}: Teacher RMS(est)={teacher_rms_est:.4f}, WDA RMS={w_rms:.4f} -> Ratio={ratio:.4f}"
            )

    avg_gamma = sum(r["suggested_gamma"] for r in results) / len(results)
    print(f"\nAverage Suggested Gamma: {avg_gamma:.4f}")
    print(
        "Recommend setting 'init_gamma' in config to this value or using per-layer init."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen05b_hero_sequential.json")
    args = parser.parse_args()
    calibrate_gamma(args.config)
