"""Evaluate wave-density model on GSM8K math reasoning benchmark."""

import re
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from wave_dencity import WaveCharLM, generate_text


def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from model generation.

    GSM8K gold answers use the format: `#### <number>`.
    For evaluation, we should ONLY trust answers explicitly marked with `####`.

    This avoids false positives like picking up "Step 5" or other incidental numbers.
    """
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def normalize_answer(ans: str) -> float | None:
    """Normalize answer to comparable float."""
    if ans is None:
        return None
    try:
        return float(ans.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def evaluate_gsm8k(
    model,
    tokenizer,
    device: str = "cuda",
    num_examples: int = 100,
    max_tokens: int = 128,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = True,
):
    """Evaluate model on GSM8K test set.
    
    Args:
        model: WaveCharLM model
        tokenizer: GPT2 tokenizer
        device: Device to run on
        num_examples: Number of test examples (max 1319 for full test set)
        max_tokens: Max generation length
        temp: Temperature for generation
        verbose: Print examples
    
    Returns:
        dict with accuracy, correct count, and examples
    """
    print(f"Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if num_examples > len(dataset):
        num_examples = len(dataset)
        print(f"Note: Using full test set ({num_examples} examples)")
    
    model.eval()
    correct = 0
    results = []
    
    print(f"\n{'='*80}")
    print(f"Evaluating on {num_examples} GSM8K problems")
    print(f"{'='*80}\n")
    
    import time
    start_time = time.time()
    
    for i, example in enumerate(dataset.select(range(num_examples))):
        iter_start = time.time()
        
        question = example['question']
        gold_answer = example['answer']
        
        # Extract ground truth answer
        gold_numeric = None
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', gold_answer)
        if match:
            gold_numeric = normalize_answer(match.group(1))
        
        # Generate model answer
        # Force a strict answer format so parsing is unambiguous.
        # This also discourages the model from emitting lots of intermediate numbers
        # that can confuse evaluation.
        prompt = (
            "You are solving a grade-school math problem. Show your reasoning briefly.End your answer with exactly:"
            "#### <integer>\n\n"
            f"Problem:\n{question}\n\nFinal answer:\n"
        )
        
        with torch.no_grad():
            generation = generate_text(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                repetition_penalty=1.15,
                device=device,
            )
        
        # Extract model's answer
        pred_answer = extract_answer(generation)
        pred_numeric = normalize_answer(pred_answer)
        
        # Check if correct
        is_correct = (gold_numeric is not None and 
                     pred_numeric is not None and 
                     abs(gold_numeric - pred_numeric) < 1e-4)
        
        if is_correct:
            correct += 1
        
        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'gold_numeric': gold_numeric,
            'generation': generation,
            'pred_answer': pred_answer,
            'pred_numeric': pred_numeric,
            'correct': is_correct,
        })
        
        # Print progress with timing
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        eta = avg_time * (num_examples - i - 1)
        
        accuracy = 100 * correct / (i + 1)
        status = '✓' if is_correct else '✗'
        
        print(f"[{i+1:3d}/{num_examples}] {status} Acc: {accuracy:5.1f}% ({correct:2d}/{i+1:2d}) | "
              f"{iter_time:5.1f}s/ex | ETA: {eta/60:.1f}m")
        
        if verbose and i < 3:  # Show first 3 examples
            print(f"  Q: {question[:100]}...")
            print(f"  Gold: {gold_numeric} | Pred: {pred_numeric}")
            print(f"  Gen: {generation[:150]}...")
            print("-" * 80)
    
    accuracy = 100 * correct / num_examples
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{num_examples})")
    print(f"{'='*80}\n")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': num_examples,
        'results': results,
    }


def main():
    import argparse
    import sys
    
    # Auto-detect best device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    
    parser = argparse.ArgumentParser(description="Evaluate wave model on GSM8K")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--num-examples", type=int, default=50, help="Number of test examples")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max generation tokens")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature")
    parser.add_argument("--verbose", action="store_true", help="Show example generations")
    parser.add_argument("--save-results", type=str, help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    # Get model config from checkpoint
    model_cfg = ckpt.get('model_cfg', {})
    if not model_cfg:
        print("ERROR: No model_cfg in checkpoint!")
        sys.exit(1)
    
    print(f"\nModel config from checkpoint:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    
    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Create model
    print("Creating model...")
    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=int(model_cfg.get('seq_len', 256)),
        embed_dim=model_cfg.get('embed_dim', 768),
        num_layers=model_cfg.get('num_layers', 8),
        num_heads=model_cfg.get('num_heads', 4),
        num_masks=model_cfg.get('num_masks', 16),
        num_waves_per_mask=model_cfg.get('num_waves_per_mask', 8),
        topk_masks=model_cfg.get('topk_masks', 8),
        attn_alpha=model_cfg.get('attn_alpha', 3.0),
        content_rank=model_cfg.get('content_rank', 8),
        content_mix=model_cfg.get('content_mix', 0.15),
        learned_content=model_cfg.get('learned_content', True),
        use_sin_waves=model_cfg.get('use_sin_waves', False),
        ffn_mult=model_cfg.get('ffn_mult', 4),
        tie_embeddings=model_cfg.get('tie_embeddings', False),
    ).to(args.device)
    
    # Load weights (strict=True since we now have matching architecture)
    print("Loading model weights...")
    try:
        # Handle torch.compile wrapper if present
        if '_orig_mod.tok_emb.weight' in ckpt['model']:
            # Checkpoint was saved after torch.compile, strip _orig_mod. prefix
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        else:
            state_dict = ckpt['model']
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"WARNING: missing={len(missing)}, unexpected={len(unexpected)}")
            if len(missing) > 0:
                print(f"  Missing keys: {missing[:5]}...")
            if len(unexpected) > 0:
                print(f"  Unexpected keys: {unexpected[:5]}...")
        else:
            print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print model info
    step = ckpt.get('step', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\n{'='*80}")
    print(f"Model: {num_params:.1f}M parameters")
    print(f"Training step: {step}")
    print(f"Best val loss: {best_val_loss:.4f} (ppl: {torch.exp(torch.tensor(best_val_loss)):.2f})")
    print(f"Device: {args.device.upper()}")
    print(f"{'='*80}\n")
    
    # Run evaluation
    results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        num_examples=args.num_examples,
        max_tokens=args.max_tokens,
        temp=args.temp,
        verbose=args.verbose,
    )
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # Don't save full results (too large), just summary + first 10 examples
            save_data = {
                'accuracy': results['accuracy'],
                'correct': results['correct'],
                'total': results['total'],
                'checkpoint': args.checkpoint,
                'model_params': num_params,
                'training_step': step,
                'best_val_loss': best_val_loss,
                'examples': results['results'][:10],
            }
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
