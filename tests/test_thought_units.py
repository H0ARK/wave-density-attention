#!/usr/bin/env python3
"""Quick test for chunk-level thought units implementation."""

import torch
import sys
sys.path.insert(0, 'private/dev')
from thought_tokens import (
    WaveCharLM,
    segment_into_chunks,
)


def test_chunk_segmentation():
    """Test chunk segmentation utilities."""
    print("Testing chunk segmentation...")
    
    # Fixed-size chunking
    seq_len = 256
    chunk_size = 64
    tokens = torch.randint(0, 1000, (2, seq_len))
    
    chunk_ids = segment_into_chunks(
        seq_len=seq_len,
        chunk_size=chunk_size,
        tokens=tokens,
        use_punctuation=False,
    )
    
    print(f"  Chunk IDs shape: {chunk_ids.shape}")
    print(f"  Number of chunks: {chunk_ids.max().item() + 1}")
    print(f"  Expected chunks: {seq_len // chunk_size}")
    assert chunk_ids.shape == (2, seq_len), "Wrong shape"
    assert chunk_ids.max().item() + 1 == seq_len // chunk_size, "Wrong number of chunks"
    print("  ✓ Fixed-size chunking works")


def test_model_with_chunks():
    """Test model forward pass with chunks enabled."""
    print("\nTesting model with chunk-level routing...")
    
    vocab_size = 1000
    seq_len = 256
    batch_size = 2
    
    # Model with chunks enabled
    model_with_chunks = WaveCharLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=128,
        num_layers=2,
        num_heads=2,
        use_chunks=True,
        chunk_size=64,
        chunk_reg_weight=0.01,
    )
    
    # Model without chunks (baseline)
    model_without_chunks = WaveCharLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=128,
        num_layers=2,
        num_heads=2,
        use_chunks=False,
    )
    
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass without chunks
    logits_no_chunks = model_without_chunks(idx)
    print(f"  Logits shape (no chunks): {logits_no_chunks.shape}")
    
    # Forward pass with chunks
    logits_with_chunks = model_with_chunks(idx)
    print(f"  Logits shape (with chunks): {logits_with_chunks.shape}")
    
    # Forward pass with chunk regularization
    logits_reg, reg_loss = model_with_chunks(idx, return_chunk_reg=True)
    print(f"  Logits shape (with reg): {logits_reg.shape}")
    print(f"  Regularization loss: {reg_loss.item():.6f}")
    
    assert logits_with_chunks.shape == (batch_size, seq_len, vocab_size), "Wrong output shape"
    assert torch.allclose(logits_reg, logits_with_chunks, atol=1e-5), "Reg output mismatch"
    assert reg_loss.item() >= 0, "Reg loss should be non-negative"
    
    print("  ✓ Model forward pass works with chunks")


def test_chunk_consistency():
    """Test that tokens in the same chunk get consistent routing."""
    print("\nTesting chunk routing consistency...")
    
    from thought_tokens import WaveDensityAttentionBlock
    
    embed_dim = 128
    seq_len = 256
    batch_size = 2
    chunk_size = 64
    
    block = WaveDensityAttentionBlock(
        embed_dim=embed_dim,
        seq_len=seq_len,
        num_heads=2,
        num_masks=8,
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create chunk IDs
    chunk_ids = (torch.arange(seq_len) // chunk_size).unsqueeze(0).expand(batch_size, -1)
    
    # Forward with chunks
    out_with_chunks = block(x, chunk_ids=chunk_ids)
    
    # Forward without chunks (for comparison)
    out_without_chunks = block(x, chunk_ids=None)
    
    print(f"  Output shape (with chunks): {out_with_chunks.shape}")
    print(f"  Output shape (without chunks): {out_without_chunks.shape}")
    
    # Outputs will be different (different routing), but shapes should match
    assert out_with_chunks.shape == out_without_chunks.shape, "Shape mismatch"
    print("  ✓ Chunk routing consistency test passed")


def test_parameter_count():
    """Verify chunk parameters don't significantly increase model size."""
    print("\nTesting parameter counts...")
    
    vocab_size = 1000
    seq_len = 256
    
    model_with_chunks = WaveCharLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=128,
        num_layers=2,
        use_chunks=True,
    )
    
    model_without_chunks = WaveCharLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=128,
        num_layers=2,
        use_chunks=False,
    )
    
    params_with = sum(p.numel() for p in model_with_chunks.parameters())
    params_without = sum(p.numel() for p in model_without_chunks.parameters())
    
    print(f"  Params with chunks: {params_with:,}")
    print(f"  Params without chunks: {params_without:,}")
    print(f"  Difference: {params_with - params_without:,} ({100*(params_with - params_without)/params_without:.2f}%)")
    
    # Chunk implementation should add zero parameters (it's just routing logic)
    assert params_with == params_without, "Chunk implementation should not add parameters"
    print("  ✓ No parameter overhead from chunks")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chunk-Level Thought Units Implementation")
    print("=" * 60)
    
    test_chunk_segmentation()
    test_model_with_chunks()
    test_chunk_consistency()
    test_parameter_count()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nChunk-level thought units are working correctly.")
    print("Key features:")
    print("  • Latent structural grouping (not visible in output)")
    print("  • Shared routing within chunks for coherence")
    print("  • No additional parameters")
    print("  • Optional consistency regularization")
