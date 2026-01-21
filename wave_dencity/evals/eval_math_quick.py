"""Quick math reasoning evaluation - pattern completion instead of full generation."""

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from wave_dencity import WaveCharLM
import sys


def eval_math_completion(model, tokenizer, device="mps"):
    """Quick math reasoning test using next-token prediction."""
    
    model.eval()
    
    # Test cases: (prompt, expected_continuation)
    test_cases = [
        # Arithmetic patterns
        ("2 + 2 =", " 4"),
        ("10 - 3 =", " 7"),
        ("5 * 6 =", " 30"),
        ("If x = 5, then x + 3 =", " 8"),
        
        # Sequences
        ("1, 2, 3, 4,", " 5"),
        ("2, 4, 6, 8,", " 10"),
        ("10, 20, 30,", " 40"),
        
        # Simple algebra
        ("x + 2 = 5, so x =", " 3"),
        ("2x = 10, so x =", " 5"),
        
        # Word problems (simplified)
        ("Alice has 3 apples. Bob gives her 2 more. She now has", " 5"),
        ("There are 10 birds. 3 fly away. Now there are", " 7"),
    ]
    
    print(f"\n{'='*80}")
    print(f"Math Reasoning Quick Eval ({len(test_cases)} tests)")
    print(f"{'='*80}\n")
    
    correct = 0
    results = []
    
    with torch.no_grad():
        for i, (prompt, expected) in enumerate(test_cases):
            # Encode prompt
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Pad to seq_len
            seq_len = model.seq_len
            if len(tokens) >= seq_len:
                idx = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
            else:
                pad_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                padding = [pad_id] * (seq_len - len(tokens))
                idx = torch.tensor([padding + tokens], dtype=torch.long, device=device)
            
            # Get next token prediction
            logits = model(idx)
            next_logits = logits[0, -1, :]  # Last position
            
            # Get top-5 predictions
            probs = F.softmax(next_logits, dim=-1)
            top5_probs, top5_ids = torch.topk(probs, 5)
            
            # Decode predictions
            predictions = []
            for prob, tok_id in zip(top5_probs, top5_ids):
                pred_text = tokenizer.decode([tok_id.item()])
                predictions.append((pred_text, prob.item()))
            
            # Check if expected is in top-5
            expected_clean = expected.strip()
            top1_pred = predictions[0][0].strip()
            
            is_correct = any(pred.strip() == expected_clean for pred, _ in predictions)
            top1_correct = top1_pred == expected_clean
            
            if top1_correct:
                correct += 1
                status = "✓"
            elif is_correct:
                status = "~"  # In top-5 but not top-1
            else:
                status = "✗"
            
            results.append({
                'prompt': prompt,
                'expected': expected,
                'predictions': predictions,
                'correct': is_correct,
                'top1_correct': top1_correct,
            })
            
            # Print result
            print(f"[{i+1:2d}/{len(test_cases)}] {status} {prompt}")
            print(f"       Expected: '{expected}'")
            print(f"       Top-1: '{predictions[0][0]}' ({predictions[0][1]*100:.1f}%)")
            if not top1_correct and is_correct:
                for j, (pred, prob) in enumerate(predictions[1:], 2):
                    if pred.strip() == expected_clean:
                        print(f"       Top-{j}: '{pred}' ({prob*100:.1f}%) ← MATCH")
                        break
            print()
    
    accuracy = 100 * correct / len(test_cases)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {correct}/{len(test_cases)} correct ({accuracy:.1f}%)")
    print(f"{'='*80}\n")
    
    return {'accuracy': accuracy, 'correct': correct, 'total': len(test_cases), 'results': results}


def main():
    import argparse
    
    # Auto-detect device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    
    parser = argparse.ArgumentParser(description="Quick math reasoning eval")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=default_device)
    
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    model_cfg = ckpt.get('model_cfg', {})
    if not model_cfg:
        print("ERROR: No model_cfg in checkpoint")
        sys.exit(1)
    
    print(f"\nModel config:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Create model
    print("Creating model...")
    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=256,
        **model_cfg
    ).to(args.device)
    
    # Load weights
    print("Loading weights...")
    if '_orig_mod.tok_emb.weight' in ckpt['model']:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    else:
        state_dict = ckpt['model']
    
    model.load_state_dict(state_dict, strict=False)
    
    step = ckpt.get('step', 0)
    val_loss = ckpt.get('best_val_loss', 0)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Step: {step}, Val PPL: {torch.exp(torch.tensor(val_loss)):.2f}")
    print(f"Device: {args.device.upper()}")
    
    # Run eval
    results = eval_math_completion(model, tokenizer, device=args.device)


if __name__ == "__main__":
    main()
