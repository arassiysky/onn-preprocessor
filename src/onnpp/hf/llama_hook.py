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

    def patched_forward(
        *args,
        **kwargs,
    ):
        """
        Supports the common HF calling patterns:
          model(input_ids=..., attention_mask=..., **etc)
          model(**tokenizer(...))

        If inputs_embeds already provided, we do not override (respect caller).
        """
        if not getattr(model, "onn_enabled", True):
            return original_forward(*args, **kwargs)

        # If caller already supplies inputs_embeds, do nothing.
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            return original_forward(*args, **kwargs)

        input_ids = kwargs.get("input_ids", None)

        # Some users pass input_ids positionally; handle simplest case:
        # If args is non-empty and kwargs has no input_ids, attempt to infer.
        if input_ids is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
            # HF forward signature varies; but first positional is usually input_ids
            input_ids = args[0]

        if input_ids is None:
            # No input_ids, nothing to do
            return original_forward(*args, **kwargs)

        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"Expected input_ids int tensor, got {input_ids.dtype}")

        # Ensure ONN is on same device as embeddings
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        # Embed using model's own embedding layer
        emb_layer = model.get_input_embeddings()

        def embed_fn(ids: torch.Tensor) -> torch.Tensor:
            return emb_layer(ids)

        # Run ONN augmentation (returns E_aug: (B,S,D), report: dict)
        E_aug, report = model.onn_preprocessor.augment_embeddings(input_ids, embed_fn=embed_fn)

        if store_last_report and model.onn_report is not None:
            model.onn_report.last = report

        # Call original forward with inputs_embeds and without input_ids
        # Preserve everything else (attention_mask, position_ids, past_key_values, labels, etc.)
        kwargs = dict(kwargs)
        kwargs.pop("input_ids", None)
        kwargs["inputs_embeds"] = E_aug

        # If args had input_ids positionally, drop it by rebuilding args with no first tensor.
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
