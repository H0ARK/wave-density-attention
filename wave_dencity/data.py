import random
import torch
from typing import Iterator

def build_streaming_dataset(tokenizer, seq_len: int = 256, buffer_size: int = 10000):
    """Build a streaming token dataset."""
    from datasets import load_dataset
    
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    def token_stream() -> Iterator[int]:
        while True:
            shuffled = dataset.shuffle(buffer_size=buffer_size, seed=None)
            for example in shuffled:
                text = example['text']
                tokens = tokenizer.encode(text, add_special_tokens=False)
                for tok in tokens:
                    yield tok
    
    return token_stream()


def build_streaming_ultrachat_dataset(
    tokenizer,
    split: str = "train_sft",
    buffer_size: int = 10000,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    assistant_only_loss: bool = True,
    include_assistant_prefix_in_loss: bool = True,
):
    """Streaming UltraChat -> stream of (token_id, loss_mask) tuples."""
    from datasets import load_dataset

    def _load_split(name: str, split_name: str):
        try:
            return load_dataset(name, split=split_name, streaming=True)
        except Exception:
            for alt in ("train", "validation", "test"):
                try:
                    return load_dataset(name, split=alt, streaming=True)
                except Exception:
                    pass
            raise

    dataset = _load_split(dataset_name, split)

    def _normalize_role(role: str) -> str:
        r = (role or "").strip().lower()
        if r in ("user", "human"): return "user"
        if r in ("assistant", "gpt", "bot"): return "assistant"
        if r == "system": return "system"
        return r or "user"

    def _extract_messages(example) -> list[dict]:
        msgs = example.get("messages")
        if isinstance(msgs, list) and msgs: return msgs
        if "prompt" in example and ("response" in example or "completion" in example):
            return [
                {"role": "user", "content": example.get("prompt", "")},
                {"role": "assistant", "content": example.get("response", example.get("completion", ""))},
            ]
        if "instruction" in example and ("output" in example or "response" in example):
            return [
                {"role": "user", "content": example.get("instruction", "")},
                {"role": "assistant", "content": example.get("output", example.get("response", ""))},
            ]
        return []

    def _encode_turn(prefix: str, content: str, loss_on_content: bool, loss_on_prefix: bool) -> tuple[list[int], list[int]]:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False) if content else []
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        ids = prefix_ids + content_ids + newline_ids
        mask = ([1 if loss_on_prefix else 0] * len(prefix_ids)) + ([1 if loss_on_content else 0] * len(content_ids)) + ([0] * len(newline_ids))
        return ids, mask

    def token_stream() -> Iterator[tuple[int, int]]:
        while True:
            shuffled = dataset.shuffle(buffer_size=buffer_size, seed=None)
            for example in shuffled:
                messages = _extract_messages(example)
                if not messages: continue
                ids_all: list[int] = []
                mask_all: list[int] = []
                for msg in messages:
                    role = _normalize_role(msg.get("role") or msg.get("from") or "user")
                    content = msg.get("content") or msg.get("value") or msg.get("text") or ""
                    content = str(content).strip()
                    if not content: continue
                    if role == "assistant":
                        prefix = "Assistant: "
                        loss_on_prefix = include_assistant_prefix_in_loss
                        loss_on_content = True
                    elif role == "system":
                        prefix = "System: "
                        loss_on_prefix = False
                        loss_on_content = False
                    else:
                        prefix = "User: "
                        loss_on_prefix = False
                        loss_on_content = False
                    ids, mask = _encode_turn(prefix, content, loss_on_content=loss_on_content, loss_on_prefix=loss_on_prefix)
                    ids_all.extend(ids)
                    mask_all.extend(mask)
                if not ids_all: continue
                if not assistant_only_loss: mask_all = [1] * len(mask_all)
                for tid, m in zip(ids_all, mask_all): yield int(tid), int(m)
    return token_stream()


def sample_batch(
    stream,
    batch_size: int,
    seq_len: int,
    device: str,
    assistant_only_loss: bool = False,
    min_supervised_tokens: int = 8,
    max_resample_tries: int = 50,
):
    """Sample a batch from the token stream."""
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        for _try in range(max_resample_tries):
            items = [next(stream) for _ in range(seq_len + 1)]
            if isinstance(items[0], tuple) or isinstance(items[0], list):
                toks = [int(t) for (t, _m) in items]
                masks = [int(_m) for (_t, _m) in items]
                supervised = sum(masks[1:])
                if (not assistant_only_loss) or supervised >= min_supervised_tokens or _try == (max_resample_tries - 1):
                    x_batch.append(toks[:-1])
                    if assistant_only_loss:
                        y = [tok if m else -100 for tok, m in zip(toks[1:], masks[1:])]
                    else:
                        y = toks[1:]
                    y_batch.append(y)
                    break
            else:
                tokens = [int(t) for t in items]
                x_batch.append(tokens[:-1])
                y_batch.append(tokens[1:])
                break
    x = torch.tensor(x_batch, dtype=torch.long, device=device)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x, y


def sample_mixed_batch(
    stream1,
    stream2,
    batch_size: int,
    seq_len: int,
    device: str,
    mix_ratio: float = 0.8,
    assistant_only_loss1: bool = True,
    assistant_only_loss2: bool = False,
    min_supervised_tokens: int = 8,
    max_resample_tries: int = 50,
):
    """Sample a mixed batch from two token streams."""
    batch_size1 = int(batch_size * mix_ratio)
    batch_size2 = batch_size - batch_size1
    x1, y1 = sample_batch(stream1, batch_size1, seq_len, device, assistant_only_loss1, min_supervised_tokens, max_resample_tries)
    x2, y2 = sample_batch(stream2, batch_size2, seq_len, device, assistant_only_loss2, min_supervised_tokens, max_resample_tries)
    x_combined = torch.cat([x1, x2], dim=0)
    y_combined = torch.cat([y1, y2], dim=0)
    indices = torch.randperm(batch_size, device=device)
    return x_combined[indices], y_combined[indices]
