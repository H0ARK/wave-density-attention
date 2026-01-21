"""Inspect checkpoint to see what's inside."""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint.pt>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT CONTENTS")
print("="*80)

for key in ckpt.keys():
    if key == 'model':
        print(f"\n{key}: (state dict with {len(ckpt[key])} keys)")
        # Show first few keys
        for i, k in enumerate(list(ckpt[key].keys())[:10]):
            print(f"  {k}: {ckpt[key][k].shape if hasattr(ckpt[key][k], 'shape') else type(ckpt[key][k])}")
        if len(ckpt[key]) > 10:
            print(f"  ... ({len(ckpt[key]) - 10} more keys)")
    elif key == 'optimizer':
        print(f"{key}: (optimizer state)")
    elif key == 'scaler':
        print(f"{key}: (grad scaler state)")
    else:
        print(f"{key}: {ckpt[key]}")

print("\n" + "="*80)
print("MODEL CONFIG")
print("="*80)

if 'model_cfg' in ckpt:
    for k, v in ckpt['model_cfg'].items():
        print(f"  {k}: {v}")
else:
    print("  No model_cfg found in checkpoint!")

print("\n" + "="*80)
print("DATA CONFIG")
print("="*80)

if 'data_cfg' in ckpt:
    for k, v in ckpt['data_cfg'].items():
        print(f"  {k}: {v}")
else:
    print("  No data_cfg found in checkpoint!")

print("\n" + "="*80)
print("TRAIN CONFIG")
print("="*80)

if 'train_cfg' in ckpt:
    for k, v in ckpt['train_cfg'].items():
        print(f"  {k}: {v}")
else:
    print("  No train_cfg found in checkpoint!")

print("\n" + "="*80)
