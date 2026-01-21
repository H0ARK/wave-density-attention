import torch
from transformers import AutoTokenizer
from wave_dencity.transplant.data import get_data_streamer
import os


def test_streamer():
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    # Use a dummy text file
    with open("test_data.txt", "w") as f:
        f.write("This is a test file for the streamer.\n" * 100)

    stream = get_data_streamer(
        tokenizer,
        dataset_cfg="test_data.txt",
        seq_len=10,
        micro_batch_size=2,
        device="cpu",
    )

    print("Getting first batch...")
    batch = next(stream)
    print(f"Batch input_ids shape: {batch.input_ids.shape}")
    print("Success!")

    os.remove("test_data.txt")


if __name__ == "__main__":
    test_streamer()
