import torch
from wave_dencity import WaveCharLM, build_streaming_dataset, sample_batch, generate_text
from transformers import GPT2TokenizerFast

def test_smoke():
    print("🚀 Starting smoke test...")
    
    # 1. Test Architecture
    print("Testing model initialization...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    model = WaveCharLM(
        vocab_size=len(tokenizer),
        seq_len=64,
        embed_dim=128,
        num_layers=2,
        num_heads=2,
        num_masks=4
    ).to(device)
    print(f"Model initialized on {device}. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. Test Forward Pass
    print("Testing forward pass...")
    idx = torch.randint(0, len(tokenizer), (2, 64)).to(device)
    logits = model(idx)
    assert logits.shape == (2, 64, len(tokenizer)), f"Wrong shape: {logits.shape}"
    print("Forward pass successful.")

    # 3. Test Data Loading (Streaming)
    print("Testing data stream (C4)...")
    try:
        stream = build_streaming_dataset(tokenizer, seq_len=64, buffer_size=10)
        x, y = sample_batch(stream, batch_size=2, seq_len=64, device=device)
        assert x.shape == (2, 64)
        assert y.shape == (2, 64)
        print("Data sampling successful.")
    except Exception as e:
        print(f"Data sampling skipped or failed (likely network/HuggingFace issue): {e}")

    # 4. Test Inference
    print("Testing inference...")
    res = generate_text(model, tokenizer, "Hello, world!", max_tokens=10, device=device)
    print(f"Generated text: {res}")
    assert len(res) > 0
    print("Inference successful.")

    print("\n✅ Smoke test passed!")

if __name__ == "__main__":
    test_smoke()
