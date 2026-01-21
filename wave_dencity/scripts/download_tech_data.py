import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_tech_data():
    target_dir = "private/data/tech_cache"
    os.makedirs(target_dir, exist_ok=True)
    
    subsets = [
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning", 15000),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Scientific-Coding", 10000),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks", 10000),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Wiki-Rewrite", 5000),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-STEM-SFT", 5000),
        ("nvidia/Nemotron-Instruction-Following-Chat-v1", "chat_if", 5000), # Chat subset to keep speech side healthy
    ]
    
    for repo, subset, num_samples in subsets:
        print(f"Downloading {num_samples} samples from {repo} ({subset})...")
        try:
            # We use streaming to pull just what we need, then save locally
            if repo == "nvidia/Nemotron-Instruction-Following-Chat-v1":
                ds = load_dataset(repo, split=subset, streaming=True)
                filename = f"chat_if.jsonl"
            else:
                ds = load_dataset(repo, subset, split="train", streaming=True)
                filename = f"{subset}.jsonl"
            
            out_path = os.path.join(target_dir, filename)
            
            with open(out_path, "w", encoding="utf-8") as f:
                for i, item in enumerate(tqdm(ds, total=num_samples)):
                    if i >= num_samples:
                        break
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  Error downloading {subset}: {e}")

if __name__ == "__main__":
    download_tech_data()
