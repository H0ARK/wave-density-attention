import torch
import argparse
import json
import hashlib
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add the repository root to the path so we can import wave_dencity
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pathlib import Path
import re
import math

from wave_dencity.transplant.adapter import load_adapter
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.wda import WDABridge


def _get_num_heads(cfg_obj) -> int:
    for key in ["num_attention_heads", "num_heads", "n_head"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_heads from model.config")


def _get_num_layers(cfg_obj) -> int:
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    raise ValueError("Could not infer num_layers from model.config")


def _layer_alpha_scale(layer_idx: int, num_layers: int, mode: str) -> float:
    if num_layers <= 1:
        return 1.0
    t = layer_idx / float(num_layers - 1)
    if mode == "uniform":
        return 1.0
    if mode == "linear":
        return float(t)
    if mode == "sqrt":
        return float(math.sqrt(max(t, 0.0)))
    return 1.0


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


def _set_scales(
    model: torch.nn.Module, alpha: float, teacher_scale: float, wda_scale: float
):
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            m.set_alpha(alpha)
            m.set_scales(teacher_scale=teacher_scale, wda_scale=wda_scale)


def generate(
    model, tokenizer, prompt, max_new_tokens=128, device="cuda", use_cache=True
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    return tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--quantization", type=str, choices=["4bit", "8bit", None], default=None
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Where to save results"
    )
    parser.add_argument(
        "--force_teacher", action="store_true", help="Force teacher run even if cached"
    )
    parser.add_argument(
        "--wda_pure_scale", type=float, default=1.0, help="Boost WDA scale in Phase 3"
    )
    parser.add_argument(
        "--skip_teacher", action="store_true", help="Skip baseline teacher run"
    )
    parser.add_argument("--skip_mixed", action="store_true", help="Skip WDA mixed run")
    parser.add_argument("--skip_pure", action="store_true", help="Skip WDA pure run")
    args = parser.parse_args()

    # Try to load config from adapter directory
    adapter_path = Path(args.adapter_path)
    config_path = adapter_path.parent / "config_used.json"
    train_cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            train_cfg = json.load(f)
        print(f"Loaded training config from {config_path}")

    model_id = args.model_id or train_cfg.get(
        "teacher_model_id", "google/gemma-3-270m-it"
    )
    quantization = args.quantization or train_cfg.get("quantization")

    # Define prompts
    prompts = [
        "If I have 15 apples, I give 3 to John. John gives 1 to Mary, and Mary gives 2 back to me. How many apples does everyone have now? Track the state step-by-step.",
        # "Write a robust Python context manager using contextlib that handles a temporary file, ensures it is deleted on exit even if an exception occurs, and logs any I/O errors.",
        # "Compare entropy in classical thermodynamics to Shannon entropy in information theory. Where exactly do the mathematical definitions overlap?",
        # "Explain what a black hole is to a second grader using a trampoline analogy.",
    ]

    # Teacher Caching logic
    teacher_cache_file = Path("teacher_cache.json")
    prompts_hash = hashlib.md5("".join(prompts).encode()).hexdigest()
    model_slug = model_id.replace("/", "_")
    cache_key = f"{model_slug}_{prompts_hash}_q{quantization}"

    teacher_results = []
    cached_data = {}
    if teacher_cache_file.exists():
        try:
            with open(teacher_cache_file, "r") as f:
                cached_data = json.load(f)
            if cache_key in cached_data and not args.force_teacher:
                teacher_results = cached_data[cache_key]
                print(f"Loaded {len(teacher_results)} teacher responses from cache.")
        except Exception as e:
            print(f"Error loading teacher cache: {e}")

    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with optional quantization
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager" if "gemma-3" in model_id.lower() else "sdpa",
        "trust_remote_code": True,
    }

    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif quantization == "8bit":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if quantization is None:
        model = model.to(args.device)
    model.eval()

    print("\n" + "=" * 50)
    print("PHASE 1: TEACHER ONLY (Baseline)")
    print("=" * 50)

    if args.skip_teacher:
        print("Skipping Phase 1 as requested.")
        if not teacher_results:
            teacher_results = ["(skipped)" for _ in prompts]
    elif not teacher_results:
        for p in prompts:
            print(f"\nPrompt: {p}")
            res = generate(model, tokenizer, p, device=args.device)
            print(f"Teacher: {res}")
            teacher_results.append(res)

        # Save to cache
        cached_data[cache_key] = teacher_results
        with open(teacher_cache_file, "w") as f:
            json.dump(cached_data, f, indent=2)
    else:
        for p, res in zip(prompts, teacher_results):
            print(f"\nPrompt: {p}")
            print(f"Teacher (cached): {res}")

    print("\n" + "=" * 50)
    print("PHASE 2: PATCHING WITH WDA ADAPTER")
    print("=" * 50)

    # Patching logic (same as training)
    hidden_size = model.config.hidden_size
    num_heads = _get_num_heads(model.config)
    num_layers = _get_num_layers(model.config)

    def make_wrapper(attn_mod: torch.nn.Module, path: str) -> torch.nn.Module:
        layer_idx = _infer_layer_index(path)
        # Match specialized_tech config: sqrt scaling
        layer_scale = _layer_alpha_scale(
            layer_idx, num_layers, train_cfg.get("alpha_layer_scale", "sqrt")
        )
        wda = WDABridge(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=train_cfg.get("seq_len", 512),
            num_masks=train_cfg.get("wda_num_masks", 32),
            num_waves_per_mask=train_cfg.get("wda_num_waves_per_mask", 4),
            topk_masks=train_cfg.get("wda_topk_masks", 8),
            attn_alpha=train_cfg.get("wda_attn_alpha", 3.0),
            content_mix=train_cfg.get("wda_content_mix", 0.1),
            learned_content=train_cfg.get("wda_learned_content", True),
            use_sin_waves=train_cfg.get("wda_use_sin_waves", True),
            use_sampling=train_cfg.get("wda_use_sampling", False),
            num_samples=train_cfg.get("wda_num_samples", 32),
            noise_sigma=train_cfg.get("wda_noise_sigma", 0.12),
            step_alpha=train_cfg.get("wda_step_alpha", 6.0),
            use_checkpoint=train_cfg.get("wda_use_checkpoint", True),
        )
        return ParallelAttentionWrapper(
            attn_mod,
            wda,
            init_alpha=0.0,
            init_teacher_scale=1.0,
            init_wda_scale=1.0,
            init_layer_alpha_scale=layer_scale,
        )

    patch_attention_modules(model, make_wrapper)
    print(f"Loading adapter from {args.adapter_path}")
    load_adapter(model, args.adapter_path, strict=False)

    # Diagnostic: Print Average Gamma
    gammas = []
    for m in model.modules():
        if isinstance(m, ParallelAttentionWrapper):
            gammas.append(m.gamma.data.cpu().float())
    if gammas:
        avg_g = torch.stack(gammas).mean().item()
        print(f"Average Gamma (Learned scale) in adapter: {avg_g:.6f}")
        if avg_g < 1e-4:
            print(
                "WARNING: Gamma is extremely low. WDA output will be effectively zeroed."
            )

    # Set scales to the evaluation alpha
    _set_scales(model, alpha=args.alpha, teacher_scale=1.0, wda_scale=1.0)
    model.eval()

    print(f"\nPhase 2: Patched with WDA Adapter (alpha={args.alpha}, ts=1.0, ws=1.0)")
    wda_mixed_results = []
    if args.skip_mixed:
        print("Skipping Phase 2 as requested.")
        wda_mixed_results = ["(skipped)" for _ in prompts]
    else:
        _set_scales(model, alpha=args.alpha, teacher_scale=1.0, wda_scale=1.0)
        for i, p in enumerate(prompts):
            print(f"\nPrompt: {p}")
            res = generate(model, tokenizer, p, device=args.device)
            print(f"WDA Mixed: {res}")
            wda_mixed_results.append(res)

    print("\n" + "=" * 50)
    print(
        f"PHASE 3: WDA BRIGDE ONLY (ts=0.0, ws={args.wda_pure_scale if hasattr(args, 'wda_pure_scale') else 5.0}, alpha=1.0)"
    )
    print("  Note: use_cache=False for speed & teacher-skip")
    print("=" * 50)
    # Force full WDA
    wda_pure_results = []
    if args.skip_pure:
        print("Skipping Phase 3 as requested.")
        wda_pure_results = ["(skipped)" for _ in prompts]
    else:
        # If it's gibberish, we likely need to boost the scale because 'gamma'
        # hasn't learned to compensate for the missing teacher yet.
        _set_scales(model, alpha=1.0, teacher_scale=0.0, wda_scale=args.wda_pure_scale)
        for i, p in enumerate(prompts):
            print(f"\nPrompt: {p}")
            # We MUST use use_cache=False here to skip the teacher's KV cache work
            res = generate(model, tokenizer, p, device=args.device, use_cache=False)
            print(f"Pure WDA: {res}")
            wda_pure_results.append(res)

    # Final saving
    output_path = args.output
    if not output_path:
        # Default to sister file of adapter
        output_path = str(adapter_path.with_suffix(".json"))

    final_output = []
    for i, p in enumerate(prompts):
        final_output.append(
            {
                "prompt": p,
                "teacher": teacher_results[i],
                "wda_mixed": wda_mixed_results[i],
                "wda_pure": wda_pure_results[i],
            }
        )

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
