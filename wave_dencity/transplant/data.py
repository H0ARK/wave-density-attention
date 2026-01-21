from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import load_dataset


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor | None = None


def stream_text_blocks(
    tokenizer,
    *,
    path: str | Path,
    seq_len: int,
    micro_batch_size: int,
    device: str,
) -> Iterator[Batch]:
    """Streams contiguous token blocks from a local text file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_path not found: {path}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def pad_or_trim(ids: list[int]) -> tuple[list[int], list[int]]:
        ids = ids[:seq_len]
        attn = [1] * len(ids)
        if len(ids) < seq_len:
            pad_n = seq_len - len(ids)
            ids = ids + [pad_id] * pad_n
            attn = attn + [0] * pad_n
        return ids, attn

    token_buffer: list[int] = []

    while True:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                token_buffer.extend(tokenizer.encode(line, add_special_tokens=False))

                while len(token_buffer) >= seq_len * micro_batch_size:
                    batch_ids = []
                    batch_attn = []
                    for _ in range(micro_batch_size):
                        chunk = token_buffer[:seq_len]
                        token_buffer = token_buffer[seq_len:]
                        ids, attn = pad_or_trim(chunk)
                        batch_ids.append(ids)
                        batch_attn.append(attn)

                    yield Batch(
                        input_ids=torch.tensor(
                            batch_ids, dtype=torch.long, device=device
                        ),
                        attention_mask=torch.tensor(
                            batch_attn, dtype=torch.long, device=device
                        ),
                    )


def _tokenize_chat_nemotron(
    tokenizer,
    item,
    seq_len: int,
    assistant_only_loss: bool,
    include_assistant_prefix_in_loss: bool = False,
):
    messages = item.get("messages")
    if messages is None:
        # Handle NVIDIA Nemotron schema if present
        if "input" in item and "output" in item:
            messages = list(item["input"])
            messages.append({"role": "assistant", "content": item["output"]})
        else:
            messages = []

    if not isinstance(messages, list) or not messages:
        return None  # Skip empty or invalid conversations

    text = None
    try:
        # Check if the template can handle this conversation
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = None

    def _simple_chat_fallback():
        ids: list[int] = []
        mask: list[int] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or content is None:
                continue
            content = str(content).strip()
            if not content:
                continue

            if role == "assistant":
                prefix = "Assistant: "
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                content_ids = tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = tokenizer.encode("\n", add_special_tokens=False)

                ids.extend(prefix_ids)
                if assistant_only_loss and not include_assistant_prefix_in_loss:
                    mask.extend([0] * len(prefix_ids))
                else:
                    mask.extend([1] * len(prefix_ids))

                ids.extend(content_ids)
                if assistant_only_loss:
                    mask.extend([1] * len(content_ids))
                else:
                    mask.extend([1] * len(content_ids))

                ids.extend(suffix_ids)
                if assistant_only_loss:
                    mask.extend([0] * len(suffix_ids))
                else:
                    mask.extend([1] * len(suffix_ids))
            elif role == "system":
                seg = f"System: {content}\n"
                seg_ids = tokenizer.encode(seg, add_special_tokens=False)
                ids.extend(seg_ids)
                if assistant_only_loss:
                    mask.extend([0] * len(seg_ids))
                else:
                    mask.extend([1] * len(seg_ids))
            else:
                seg = f"User: {content}\n"
                seg_ids = tokenizer.encode(seg, add_special_tokens=False)
                ids.extend(seg_ids)
                if assistant_only_loss:
                    mask.extend([0] * len(seg_ids))
                else:
                    mask.extend([1] * len(seg_ids))
        if not ids:
            return None
        return ids, mask

    if text is None:
        res = _simple_chat_fallback()
        return res

    enc = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"]

    if not assistant_only_loss:
        return input_ids, [1] * len(input_ids)

    loss_mask = [0] * len(input_ids)

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if not role or content is None:
            continue

        # Get text for everything BEFORE this message
        try:
            if i == 0:
                prefix_text = ""
            else:
                prefix_text = tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=False
                )

            # Get text INCLUDING this message
            full_msg_text = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
        except Exception:
            continue

        p_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        m_ids = tokenizer.encode(full_msg_text, add_special_tokens=False)

        # Message tokens (approximate if tokenizer merges across boundaries)
        # But for chat templates, boundaries are usually clean tags.
        start_idx = len(p_ids)
        end_idx = len(m_ids)

        if role == "assistant":
            if include_assistant_prefix_in_loss:
                # Mask everything in this message turn
                for j in range(start_idx, end_idx):
                    if j < len(loss_mask):
                        loss_mask[j] = 1
            else:
                # Mask only the CONTENT, skip the header/footer tags
                # We find the content's tokens within the message tokens
                c_ids = tokenizer.encode(content, add_special_tokens=False)
                # Heuristic: find where c_ids appears in m_ids[start_idx:]
                found = False
                for j in range(start_idx, end_idx - len(c_ids) + 1):
                    if m_ids[j : j + len(c_ids)] == c_ids:
                        for k in range(j, j + len(c_ids)):
                            if k < len(loss_mask):
                                loss_mask[k] = 1
                        found = True
                        break
                if not found:
                    # Fallback to full message if content tokens don't match exactly
                    for j in range(start_idx, end_idx):
                        if j < len(loss_mask):
                            loss_mask[j] = 1

    return input_ids, loss_mask


def stream_dataset(
    tokenizer,
    cfg: str | dict[str, Any],
    seq_len: int,
    micro_batch_size: int,
    device: str,
) -> Iterator[Batch]:
    if isinstance(cfg, str):
        yield from stream_text_blocks(
            tokenizer,
            path=cfg,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        return

    kind = cfg.get("kind", "text")
    if kind == "text":
        path = cfg.get("path", cfg.get("dataset_path"))
        if not path:
            raise ValueError(
                "Dataset kind 'text' requires 'path' or 'dataset_path' key"
            )
        yield from stream_text_blocks(
            tokenizer,
            path=path,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            device=device,
        )
    elif kind == "hf_chat" or kind == "nemotron_chat":
        ds_name = cfg["dataset_name"]
        subset = cfg.get("subset")
        split = cfg["split"]
        assistant_only_loss = cfg.get("assistant_only_loss", True)
        include_assistant_prefix_in_loss = cfg.get(
            "include_assistant_prefix_in_loss", False
        )

        token_buffer = []
        mask_buffer = []

        while True:
            # subset=None is handled by load_dataset
            ds = load_dataset(ds_name, subset, split=split, streaming=True)
            for item in ds:
                res = _tokenize_chat_nemotron(
                    tokenizer,
                    item,
                    seq_len,
                    assistant_only_loss,
                    include_assistant_prefix_in_loss=include_assistant_prefix_in_loss,
                )
                if res is None:
                    continue
                ids, mask = res
                token_buffer.extend(ids)
                mask_buffer.extend(mask)

                # Use a larger buffer to yield full micro-batches
                while len(token_buffer) >= (seq_len * micro_batch_size):
                    batch_ids = []
                    batch_attn = []
                    batch_loss_mask = []

                    for _ in range(micro_batch_size):
                        chunk_ids = token_buffer[:seq_len]
                        chunk_mask = mask_buffer[:seq_len]
                        token_buffer = token_buffer[seq_len:]
                        mask_buffer = mask_buffer[seq_len:]

                        batch_ids.append(chunk_ids)
                        batch_attn.append([1] * seq_len)
                        batch_loss_mask.append(chunk_mask)

                    yield Batch(
                        input_ids=torch.tensor(
                            batch_ids, dtype=torch.long, device=device
                        ),
                        attention_mask=torch.tensor(
                            batch_attn, dtype=torch.long, device=device
                        ),
                        loss_mask=torch.tensor(
                            batch_loss_mask, dtype=torch.float32, device=device
                        ),
                    )
    elif kind == "hf_text":
        ds_name = cfg["dataset_name"]
        split = cfg.get("split", "train")
        subset = cfg.get("subset")
        text_column = cfg.get("text_column", "text")

        token_buffer = []
        while True:
            # use_auth_token=True can sometimes help with NVIDIA datasets
            ds = load_dataset(ds_name, subset, split=split, streaming=True)
            for item in ds:
                text = item.get(text_column)
                if not text or not isinstance(text, str):
                    continue

                # To prevent memory spikes on Windows with massive files,
                # we process the text in large character chunks
                chunk_len = 100000
                for i in range(0, len(text), chunk_len):
                    subtext = text[i : i + chunk_len]
                    ids = tokenizer.encode(subtext, add_special_tokens=False)
                    token_buffer.extend(ids)

                    # Only process tokens IF we have enough, to keep yield frequency stable
                    while len(token_buffer) >= (seq_len * micro_batch_size):
                        batch_ids = []
                        batch_attn = []
                        for _ in range(micro_batch_size):
                            chunk_ids = token_buffer[:seq_len]
                            token_buffer = token_buffer[seq_len:]
                            batch_ids.append(chunk_ids)
                            batch_attn.append([1] * seq_len)

                        yield Batch(
                            input_ids=torch.tensor(
                                batch_ids, dtype=torch.long, device=device
                            ),
                            attention_mask=torch.tensor(
                                batch_attn, dtype=torch.long, device=device
                            ),
                            loss_mask=None,
                        )

                # Add EOS at document boundary
                if tokenizer.eos_token_id is not None:
                    token_buffer.append(tokenizer.eos_token_id)
    elif kind == "local_jsonl":
        path = cfg["path"]
        is_chat = cfg.get("is_chat", False)
        assistant_only_loss = cfg.get("assistant_only_loss", True)
        include_assistant_prefix_in_loss = cfg.get(
            "include_assistant_prefix_in_loss", False
        )
        text_column = cfg.get("text_column", "text")

        assistant_prefixes = cfg.get("assistant_prefixes", None)
        if assistant_prefixes is None:
            assistant_prefix = cfg.get("assistant_prefix", "Assistant:")
            assistant_prefixes = [assistant_prefix]
        if isinstance(assistant_prefixes, str):
            assistant_prefixes = [assistant_prefixes]
        assistant_prefixes = [p for p in assistant_prefixes if p]
        assistant_only_fallback = cfg.get("assistant_only_fallback", "all")

        token_buffer = []
        mask_buffer = []

        while True:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if is_chat:
                        res = _tokenize_chat_nemotron(
                            tokenizer,
                            item,
                            seq_len,
                            assistant_only_loss,
                            include_assistant_prefix_in_loss,
                        )
                        if res is None:
                            continue
                        ids, mask = res
                    else:
                        text = item.get(text_column, "")
                        if not text:
                            continue
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        ids.append(tokenizer.eos_token_id or 0)

                        if assistant_only_loss and assistant_prefixes:
                            idx = -1
                            marker = None
                            for p in assistant_prefixes:
                                idx = text.find(p)
                                if idx >= 0:
                                    marker = p
                                    break

                            if idx >= 0 and marker is not None:
                                prefix_text = text[:idx]
                                if include_assistant_prefix_in_loss:
                                    start = len(
                                        tokenizer.encode(
                                            prefix_text, add_special_tokens=False
                                        )
                                    )
                                else:
                                    start = len(
                                        tokenizer.encode(
                                            prefix_text + marker,
                                            add_special_tokens=False,
                                        )
                                    )
                                mask = [0] * len(ids)
                                for i in range(start, len(ids)):
                                    mask[i] = 1
                            else:
                                if assistant_only_fallback == "skip":
                                    continue
                                mask = [1] * len(ids)
                        else:
                            mask = [1] * len(ids)

                    token_buffer.extend(ids)
                    mask_buffer.extend(mask)

                    # Only yield when we can fill an ENTIRE micro-batch.
                    # This reduces the overhead of small yields.
                    needed = seq_len * micro_batch_size
                    if len(token_buffer) >= needed:
                        batch_ids = []
                        batch_attn = []
                        batch_loss_mask = []
                        for _ in range(micro_batch_size):
                            chunk_ids = token_buffer[:seq_len]
                            chunk_mask = mask_buffer[:seq_len]
                            token_buffer = token_buffer[seq_len:]
                            mask_buffer = mask_buffer[seq_len:]

                            batch_ids.append(chunk_ids)
                            batch_attn.append([1] * seq_len)
                            batch_loss_mask.append(chunk_mask)

                        yield Batch(
                            input_ids=torch.tensor(
                                batch_ids, dtype=torch.long, device=device
                            ),
                            attention_mask=torch.tensor(
                                batch_attn, dtype=torch.long, device=device
                            ),
                            loss_mask=torch.tensor(
                                batch_loss_mask, dtype=torch.float32, device=device
                            ),
                        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")


def get_data_streamer(
    tokenizer,
    dataset_cfg: str | list[dict[str, Any]],
    seq_len: int,
    micro_batch_size: int,
    device: str,
) -> Iterator[Batch]:
    if isinstance(dataset_cfg, str):
        yield from stream_text_blocks(
            tokenizer,
            path=dataset_cfg,
            seq_len=seq_len,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        return

    if len(dataset_cfg) == 1:
        yield from stream_dataset(
            tokenizer, dataset_cfg[0], seq_len, micro_batch_size, device
        )
        return

    streamers = [
        stream_dataset(tokenizer, cfg, seq_len, micro_batch_size, device)
        for cfg in dataset_cfg
    ]
    ratios = [cfg.get("ratio", 1.0) for cfg in dataset_cfg]
    total = sum(ratios)
    probs = [r / total for r in ratios]

    while True:
        s_idx = random.choices(range(len(streamers)), weights=probs, k=1)[0]
        try:
            yield next(streamers[s_idx])
        except StopIteration:
            streamers[s_idx] = stream_dataset(
                tokenizer, dataset_cfg[s_idx], seq_len, micro_batch_size, device
            )
            yield next(streamers[s_idx])
