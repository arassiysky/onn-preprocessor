from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from onnpp.hf.utils import get_embedding_device, move_module_to_device


class ONNHookReport:
    """
    Lightweight container for last ONN report (optional).
    """
    def __init__(self):
        self.last: Optional[Dict[str, Any]] = None


def wrap_llama_for_onn(model: torch.nn.Module, onn, *, store_last_report: bool = True):
    """
    Monkey-patches a HF causal LM model so that when users call model(input_ids=...),
    we intercept embeddings, run ONN augmentation (v0.L), and forward using inputs_embeds.

    Requirements:
      - model.get_input_embeddings() exists
      - model.forward accepts inputs_embeds (true for HF Llama-like models)

    Returns:
      model (same object), patched in-place.

    After wrapping, you can toggle ONN at runtime:
      model.onn_enabled = True/False

    If store_last_report=True, last report is stored at:
      model.onn_report.last
    """
    if not hasattr(model, "forward"):
        raise ValueError("Model has no forward()")

    if not hasattr(model, "get_input_embeddings"):
        raise ValueError("Model has no get_input_embeddings(); not a HF-style model?")

    original_forward = model.forward

    # Attach ONN and control flags to model
    model.onn_preprocessor = onn
    model.onn_enabled = True
    model.onn_report = ONNHookReport() if store_last_report else None

    # Move ONN to model embedding device (important on GPU / multi-device setups)
    device = get_embedding_device(model)
    move_module_to_device(model.onn_preprocessor, device)

    def patched_forward(*args, **kwargs):
        if not getattr(model, "onn_enabled", True):
            return original_forward(*args, **kwargs)

        # Respect caller-provided inputs_embeds
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            return original_forward(*args, **kwargs)

        # Pull input_ids from kwargs or first positional tensor
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]

        if input_ids is None:
            return original_forward(*args, **kwargs)

        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"Expected input_ids int tensor, got {input_ids.dtype}")

        # Ensure device match
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        # Attention mask (optional)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        # ---- Option 1: optional compression ----
        comp_rep = None
        if getattr(model.onn_preprocessor.config, "compression", "none") != "none":
            input_ids, attention_mask, comp_rep = model.onn_preprocessor.compress_tokens(input_ids, attention_mask)

            # overwrite attention_mask in kwargs for downstream
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = attention_mask

        # ---- Option 2: embedding augmentation ----
        emb_layer = model.get_input_embeddings()

        def embed_fn(ids: torch.Tensor) -> torch.Tensor:
            return emb_layer(ids)

        E_aug, report = model.onn_preprocessor.augment_embeddings(input_ids, embed_fn=embed_fn)

        # Attach compression info to the same report dict (if any)
        if comp_rep is not None:
            report["compression"] = comp_rep

        # Store report safely
        if store_last_report and model.onn_report is not None:
            model.onn_report.last = report

        # Forward using inputs_embeds; remove input_ids (and drop positional if present)
        kwargs = dict(kwargs)
        kwargs.pop("input_ids", None)
        kwargs["inputs_embeds"] = E_aug

        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            args = args[1:]

        return original_forward(*args, **kwargs)


    # Patch
    model.forward = patched_forward  # type: ignore[assignment]
    model._onn_original_forward = original_forward  # for unwrapping/debug

    return model


def unwrap_llama_from_onn(model: torch.nn.Module):
    """
    Restores original forward if the model was wrapped.
    """
    orig = getattr(model, "_onn_original_forward", None)
    if orig is None:
        return model
    model.forward = orig  # type: ignore[assignment]
    # clean up
    for attr in ("onn_preprocessor", "onn_enabled", "onn_report", "_onn_original_forward"):
        if hasattr(model, attr):
            delattr(model, attr)
    return model
