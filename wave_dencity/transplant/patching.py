from __future__ import annotations

import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class ModuleRef:
    path: str
    parent: nn.Module
    name: str
    module: nn.Module


def _iter_module_refs(root: nn.Module, prefix: str = "") -> Iterable[ModuleRef]:
    for name, child in root._modules.items():
        if child is None:
            continue
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        yield ModuleRef(path=path, parent=root, name=name, module=child)
        yield from _iter_module_refs(child, prefix=path)


def find_attention_modules(model: nn.Module) -> list[ModuleRef]:
    """Heuristic scan for per-layer self-attention modules in HF decoder stacks."""
    candidates: list[ModuleRef] = []

    # Common attribute names for decoder layer attention
    attn_attr_names = {"self_attn", "self_attention", "attn", "attention"}

    for ref in _iter_module_refs(model):
        if ref.name in attn_attr_names:
            candidates.append(ref)
            continue
        # Fallback: modules whose class name smells like attention
        cls = ref.module.__class__.__name__.lower()
        if "attention" in cls and (ref.name.endswith("attn") or ref.name.endswith("attention")):
            candidates.append(ref)

    # Deduplicate by path
    uniq: dict[str, ModuleRef] = {r.path: r for r in candidates}
    return [uniq[k] for k in sorted(uniq.keys())]


def patch_attention_modules(
    model: nn.Module,
    make_wrapper: Callable[[nn.Module, str], nn.Module],
    *,
    filter_paths: Callable[[str], bool] | None = None,
) -> list[str]:
    """Replace attention modules in-place; returns patched module paths."""
    patched: list[str] = []
    for ref in find_attention_modules(model):
        if filter_paths is not None and not filter_paths(ref.path):
            continue
        wrapper = make_wrapper(ref.module, ref.path)
        setattr(ref.parent, ref.name, wrapper)
        patched.append(ref.path)
    return patched
