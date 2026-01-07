# Wave-Density Attention

Wave-Density Attention (WDA) is an alternative to dot-product self-attention where attention structure emerges from **wave interference + density formation**, rather than explicit $QK^T$ similarity.

## Repository Structure

- `wave_dencity/`: Core Python package containing the model, data loaders, and inference code.
- `train.py`: Main entry point for training a model on C4 or UltraChat.
- `evals/`: Scripts for evaluating models on GSM8K, MATH, and perplexity.
- `scripts/`: Utility scripts for checkpoint inspection, interaction, and demo.
- `paper/`: LaTeX source for the technical paper.

## Quickstart

### Install

```bash
pip install -e .
```

### Train (streaming from Hugging Face)

```bash
python3 train.py
```

By default, the training code streams datasets via `datasets` (so large corpora are **not** stored in this repo).

### Evaluate a checkpoint

```bash
python3 evals/evaluate_model.py --checkpoint /path/to/checkpoint.pt
```

### Quick perplexity probe

```bash
python3 evals/quick_ppl.py /path/to/checkpoint.pt --dataset ultrachat --batches 10 --batch-size 32
```

## Paper

- Source: `paper/main.tex`
- Build: see `paper/README.md`

## What is *not* in git

Large artifacts (checkpoints, downloaded corpora, token caches) should live in `private/`.
That folder is intentionally ignored by `.gitignore`.

## License

Apache-2.0 (see `LICENSE`).
