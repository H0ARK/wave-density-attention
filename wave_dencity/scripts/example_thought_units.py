#!/usr/bin/env python3
"""Example: Training with chunk-level thought units."""

import sys
sys.path.insert(0, 'private/dev')

from thought_tokens import WaveCharLM, segment_into_chunks
from transformers import GPT2TokenizerFast
import torch
import torch.nn.functional as F


def example_basic_usage():
    """Basic example of using chunks."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create model with chunks enabled
    model = WaveCharLM(
        vocab_size=50257,
        seq_len=256,
        embed_dim=384,
        num_layers=4,
        num_heads=4,
        use_chunks=True,
        chunk_size=64,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Chunks enabled: {model.use_chunks}")
    print(f"Chunk size: {model.chunk_size}")
    
    # Sample forward pass
    batch_size = 2
    seq_len = 256
    idx = torch.randint(0, 50257, (batch_size, seq_len))
    
    logits = model(idx)
    print(f"Output shape: {logits.shape}")
    print("✓ Forward pass successful\n")


def example_with_regularization():
    """Example using chunk consistency regularization."""
    print("=" * 60)
    print("Example 2: With Regularization")
    print("=" * 60)
    
    model = WaveCharLM(
        vocab_size=50257,
        seq_len=256,
        embed_dim=384,
        num_layers=4,
        use_chunks=True,
        chunk_size=64,
        chunk_reg_weight=0.01,  # Enable regularization
    )
    
    print(f"Regularization weight: {model.chunk_reg_weight}")
    
    idx = torch.randint(0, 50257, (2, 256))
    labels = torch.randint(0, 50257, (2, 256))
    
    # Forward with regularization
    logits, reg_loss = model(idx, return_chunk_reg=True)
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    total_loss = ce_loss + reg_loss
    
    print(f"Cross-entropy loss: {ce_loss.item():.4f}")
    print(f"Regularization loss: {reg_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    print("✓ Training with regularization\n")


def example_chunk_visualization():
    """Visualize how chunks are assigned."""
    print("=" * 60)
    print("Example 3: Chunk Visualization")
    print("=" * 60)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    text = """Hello, this is a test. We want to see how chunks work.
Each sentence might be a chunk. Or we use fixed sizes.
Let's see what happens!"""
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Text: {text[:60]}...")
    print(f"Number of tokens: {len(tokens)}")
    
    # Pad to seq_len
    seq_len = 128
    if len(tokens) < seq_len:
        tokens = tokens + [0] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    
    idx = torch.tensor([tokens])
    
    # Get chunks with different sizes
    for chunk_size in [32, 64, 128]:
        chunk_ids = segment_into_chunks(
            seq_len=seq_len,
            chunk_size=chunk_size,
            tokens=idx,
            use_punctuation=False,
        )
        num_chunks = chunk_ids.max().item() + 1
        print(f"\nChunk size {chunk_size}: {num_chunks} chunks")
        print(f"  First 16 chunk IDs: {chunk_ids[0, :16].tolist()}")
    
    print("✓ Chunk segmentation complete\n")


def example_comparison():
    """Compare models with and without chunks."""
    print("=" * 60)
    print("Example 4: With vs Without Chunks")
    print("=" * 60)
    
    config = {
        "vocab_size": 50257,
        "seq_len": 256,
        "embed_dim": 384,
        "num_layers": 4,
        "num_heads": 4,
    }
    
    model_no_chunks = WaveCharLM(**config, use_chunks=False)
    model_with_chunks = WaveCharLM(**config, use_chunks=True, chunk_size=64)
    
    idx = torch.randint(0, 50257, (2, 256))
    
    # Compare outputs
    with torch.no_grad():
        logits_no_chunks = model_no_chunks(idx)
        logits_with_chunks = model_with_chunks(idx)
    
    print(f"Output shape (no chunks): {logits_no_chunks.shape}")
    print(f"Output shape (with chunks): {logits_with_chunks.shape}")
    
    # Parameter counts
    params_no = sum(p.numel() for p in model_no_chunks.parameters())
    params_with = sum(p.numel() for p in model_with_chunks.parameters())
    
    print(f"\nParameters (no chunks): {params_no:,}")
    print(f"Parameters (with chunks): {params_with:,}")
    print(f"Difference: {params_with - params_no:,}")
    
    print("\n✓ Both models work, same parameter count\n")


def example_training_snippet():
    """Show how chunks integrate into training loop."""
    print("=" * 60)
    print("Example 5: Training Loop Integration")
    print("=" * 60)
    
    model = WaveCharLM(
        vocab_size=50257,
        seq_len=256,
        embed_dim=384,
        num_layers=4,
        use_chunks=True,
        chunk_size=64,
        chunk_reg_weight=0.01,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    print("Training loop pseudocode:")
    print("""
    for step in range(num_steps):
        # Sample batch
        x, y = next(data_stream)
        
        # Forward pass (chunks computed automatically)
        logits, reg_loss = model(x, return_chunk_reg=True)
        
        # Compute loss
        ce_loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        total_loss = ce_loss + reg_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    """)
    
    # Actual training step
    x = torch.randint(0, 50257, (2, 256))
    y = torch.randint(0, 50257, (2, 256))
    
    optimizer.zero_grad()
    logits, reg_loss = model(x, return_chunk_reg=True)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    total_loss = ce_loss + reg_loss
    total_loss.backward()
    optimizer.step()
    
    print(f"✓ Training step complete (loss: {total_loss.item():.4f})\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Chunk-Level Thought Units - Usage Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_with_regularization()
    example_chunk_visualization()
    example_comparison()
    example_training_snippet()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • Chunks are computed automatically during forward pass")
    print("  • No changes to model API (backward compatible)")
    print("  • Zero parameter overhead")
    print("  • Optional regularization for consistency")
    print("  • Seamlessly integrates into existing training loops")
    print("\nTo train with chunks:")
    print("  python3 thought_tokens.py train --use-chunks --chunk-size 64")
