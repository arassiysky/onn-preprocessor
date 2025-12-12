from __future__ import annotations

from typing import Optional

import torch


def get_hidden_size_from_model(model) -> int:
    """
    Best-effort to fetch d_model / hidden_size from HF model.config.
    Works for Llama, Mistral, etc.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no .config; cannot infer hidden size.")

    for attr in ("hidden_size", "n_embd", "d_model", "dim"):
        if hasattr(cfg, attr):
            val = int(getattr(cfg, attr))
            if val > 0:
                return val

    raise ValueError("Could not infer hidden size from model.config (tried hidden_size, n_embd, d_model, dim).")


def move_module_to_device(module: torch.nn.Module, device: torch.device) -> None:
    """
    Safely move module parameters/buffers to device.
    """
    module.to(device)


def get_embedding_device(model) -> torch.device:
    """
    Returns the device of model input embeddings.
    """
    emb = model.get_input_embeddings()
    if emb is None:
        # Fallback: try first parameter device
        return next(model.parameters()).device
    return emb.weight.device
