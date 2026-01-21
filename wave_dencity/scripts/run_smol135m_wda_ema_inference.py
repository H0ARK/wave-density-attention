import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running this script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_dencity.transplant.adapter import (
    load_adapter,
    offload_teacher_attention,
    set_parallel_attn_mode,
)
from wave_dencity.transplant.patching import patch_attention_modules
from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper
from wave_dencity.transplant.wda import WDABridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument(
        "--adapter",
        default="private/checkpoints/smol135m_wda/adapter_final_ema.pt",
        help="Path to adapter_*.pt produced by training",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--prompt", default="Explain wave-density attention in one paragraph."
    )
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument(
        "--use_teacher_forward",
        action="store_true",
        help="Keep teacher_scale>0 (slower, needs teacher_attn on GPU)",
    )
    args = ap.parse_args()

    cfg_path = Path("configs/transplant_smol135m.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    device = args.device
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)

    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    )
    num_heads = int(
        getattr(
            model.config, "num_attention_heads", getattr(model.config, "n_head", None)
        )
    )

    def make_wrapper(attn_mod, path):
        wda = WDABridge(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=cfg["seq_len"],
            num_masks=cfg["wda_num_masks"],
            num_waves_per_mask=cfg["wda_num_waves_per_mask"],
            topk_masks=cfg["wda_topk_masks"],
            gate_temp=cfg.get("wda_gate_temp", 0.05),
            attn_alpha=cfg["wda_attn_alpha"],
            content_mix=cfg.get("wda_content_mix", 0.15),
            learned_content=cfg.get("wda_learned_content", True),
            use_sin_waves=cfg.get("wda_use_sin_waves", True),
            noise_sigma=cfg.get("wda_noise_sigma", 0.12),
            step_alpha=cfg.get("wda_step_alpha", 6.0),
        )
        return ParallelAttentionWrapper(
            attn_mod, wda, init_alpha=1.0, init_teacher_scale=0.0, init_wda_scale=1.0
        )

    patch_attention_modules(model, make_wrapper)

    # Load trained weights
    load_adapter(model, args.adapter, strict=False)

    # WDA-only cacheless mode by default
    if args.use_teacher_forward:
        # Keep a teacher tether (will require teacher_attn on GPU)
        set_parallel_attn_mode(
            model,
            alpha=0.995,
            teacher_scale=0.07,
            wda_scale=1.0,
            gate_temp=cfg.get("wda_gate_temp", 0.05),
        )
    else:
        set_parallel_attn_mode(
            model,
            alpha=1.0,
            teacher_scale=0.0,
            wda_scale=1.0,
            gate_temp=cfg.get("wda_gate_temp", 0.05),
        )
        # Offload frozen teacher attention branch to CPU to reclaim VRAM.
        offload_teacher_attention(model, device="cpu")
        # Ensure generation does not request KV-cache (teacher_scale=0 forbids it)
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False

    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=False,
        )

    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
