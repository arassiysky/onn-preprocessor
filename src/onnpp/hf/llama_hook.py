# llama_hook.py
from __future__ import annotations

from typing import Any, Dict, Optional

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
    we intercept embeddings, run ONN compression (optional) + embedding augmentation, and
    forward using inputs_embeds.

    IMPORTANT GENERATION NOTE
    -------------------------
    HF `generate()` uses KV-cache + rotary position embeddings (RoPE). If we change the
    effective sequence length mid-generation (i.e. by compressing input_ids during cached
    decode steps), RoPE/cache_position can become inconsistent and crash.

    Therefore this hook will:
      - allow compression on the initial prompt prefill (no cache)
      - automatically DISABLE compression whenever `past_key_values` or `cache_position`
        are present (cached decode steps)
    Embedding augmentation remains enabled for all steps.

    L2/L4 WIRING
    ------------
    If the ONNPreprocessor.compress_tokens() supports L2/L4 (new preprocessor.py),
    we pass `model=` and `task_mode=` through so L2 can read runtime flags like:
      - model.onn_is_sampling (set by benchmark before generate() branches)
      - model.onn_l2_enabled / model.onn_l4_enabled (optional toggles)
      - model.onn_l2_* overrides (optional)

    Requirements:
      - model.get_input_embeddings() exists
      - model.forward accepts inputs_embeds (true for HF Llama-like models)

    Returns:
      model (same object), patched in-place.

    Runtime controls (safe defaults for inference):
      model.onn_enabled = True/False
      model.onn_task_mode = "text" or "code"          (default: "text")
      model.onn_disable_compression_for_code = True   (default: True)
      model.onn_report_every = N                      (default: 1; 0 disables)
      model.onn_use_no_grad_in_inference = True       (default: True)

    Additional runtime hints (optional):
      model.onn_is_sampling = True/False              (helps L2 clamp quantile)
      model.onn_l2_enabled = True/False
      model.onn_l4_enabled = True/False

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
    model.onn_task_mode = "code"
    model.onn_disable_compression_for_code = False
    model.onn_l2_min_len_enable = 128                # REQUIRED for tgt=128
    model.onn_l2_enabled = True
    model.onn_l4_enabled = True
    model.onn_report_every = 1
    model.onn_use_no_grad_in_inference = True

    # Optional runtime hints (bench/scripts can set these)
    if not hasattr(model, "onn_is_sampling"):
        model.onn_is_sampling = False  # type: ignore[attr-defined]
    if not hasattr(model, "onn_l2_enabled"):
        model.onn_l2_enabled = True  # type: ignore[attr-defined]
    if not hasattr(model, "onn_l4_enabled"):
        model.onn_l4_enabled = True  # type: ignore[attr-defined]

    model.onn_report = ONNHookReport() if store_last_report else None

    # Move ONN to current embedding device; also cache for later device migration
    dev0 = get_embedding_device(model)
    move_module_to_device(model.onn_preprocessor, dev0)
    model._onn_last_device = dev0  # type: ignore[attr-defined]
    model._onn_step = 0  # type: ignore[attr-defined]

    def _maybe_move_onn_to_device(dev: torch.device):
        last = getattr(model, "_onn_last_device", None)
        if last is None or last != dev:
            move_module_to_device(model.onn_preprocessor, dev)
            model._onn_last_device = dev  # type: ignore[attr-defined]

    def patched_forward(*args, **kwargs):
        # If ONN is disabled, behave exactly like the original model
        if not getattr(model, "onn_enabled", True):
            return original_forward(*args, **kwargs)

        # Respect caller-provided inputs_embeds (do not override)
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

        # Re-check embedding device in case user moved model after wrapping
        device = get_embedding_device(model)
        _maybe_move_onn_to_device(device)

        # Ensure device match (use non_blocking where safe)
        if input_ids.device != device:
            input_ids = input_ids.to(device, non_blocking=True)

        # Attention mask (optional)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device, non_blocking=True)

        # Detect cached decode step inside HF generate()
        # - past_key_values present => cache is active
        # - cache_position present (newer HF) => cache position tracking active
        is_cached_decode = (kwargs.get("past_key_values", None) is not None) or (
            kwargs.get("cache_position", None) is not None
        )

        # ---- Optional compression ----
        comp_rep = None
        mode = getattr(model, "onn_task_mode", "text")
        disable_comp_for_code = getattr(model, "onn_disable_compression_for_code", True)

        cfg = getattr(getattr(model, "onn_preprocessor", None), "config", None)
        compression_mode = getattr(cfg, "compression", "none") if cfg is not None else "none"

        do_compress = (compression_mode != "none")

        # Safe default: disable compression for code unless explicitly forced
        if mode == "code" and disable_comp_for_code:
            do_compress = False

        # CRITICAL: disable compression during cached decode steps (generation)
        if is_cached_decode:
            do_compress = False

        if do_compress:
            # L2/L4 wiring: pass model + task_mode if supported (new preprocessor.py)
            try:
                input_ids, attention_mask, comp_rep = model.onn_preprocessor.compress_tokens(
                    input_ids,
                    attention_mask,
                    model=model,
                    task_mode=mode,
                    # is_sampling optional; preprocessor will read model.onn_is_sampling if not provided
                )
            except TypeError:
                # Back-compat with older preprocessor signatures
                input_ids, attention_mask, comp_rep = model.onn_preprocessor.compress_tokens(
                    input_ids,
                    attention_mask,
                )

            # overwrite attention_mask in kwargs for downstream
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = attention_mask

            # If caller provided position_ids, they may now be inconsistent with compressed length.
            # For safety, drop them so the model can infer positions from attention_mask/length.
            kwargs.pop("position_ids", None)

        # ---- Embedding augmentation ----
        emb_layer = model.get_input_embeddings()

        def embed_fn(ids: torch.Tensor) -> torch.Tensor:
            return emb_layer(ids)

        # Inference-only: use no_grad if gradients are disabled and flag is enabled
        use_no_grad = getattr(model, "onn_use_no_grad_in_inference", True) and (not torch.is_grad_enabled())

        if use_no_grad:
            with torch.no_grad():
                E_aug, report = model.onn_preprocessor.augment_embeddings(input_ids, embed_fn=embed_fn)
        else:
            E_aug, report = model.onn_preprocessor.augment_embeddings(input_ids, embed_fn=embed_fn)

        # Attach compression info to report dict (if any)
        if isinstance(report, dict):
            report["task_mode"] = mode
            report["cached_decode"] = bool(is_cached_decode)
            report["is_sampling"] = bool(getattr(model, "onn_is_sampling", False))
            report["l2_enabled"] = bool(getattr(model, "onn_l2_enabled", True))
            report["l4_enabled"] = bool(getattr(model, "onn_l4_enabled", True))
            if comp_rep is not None:
                report["compression"] = comp_rep

        # Store report cheaply (optional + throttled)
        if store_last_report and model.onn_report is not None:
            every = int(getattr(model, "onn_report_every", 1) or 0)
            step = int(getattr(model, "_onn_step", 0))
            if every > 0 and (step % every == 0):
                model.onn_report.last = report
            model._onn_step = step + 1  # type: ignore[attr-defined]

        # Forward using inputs_embeds; remove input_ids (and drop positional if present)
        kwargs = dict(kwargs)
        kwargs.pop("input_ids", None)
        kwargs["inputs_embeds"] = E_aug

        # If first positional arg was input_ids tensor, drop it
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
    for attr in (
        "onn_preprocessor",
        "onn_enabled",
        "onn_task_mode",
        "onn_disable_compression_for_code",
        "onn_report_every",
        "onn_use_no_grad_in_inference",
        "onn_is_sampling",
        "onn_l2_enabled",
        "onn_l4_enabled",
        "onn_report",
        "_onn_original_forward",
        "_onn_last_device",
        "_onn_step",
    ):
        if hasattr(model, attr):
            delattr(model, attr)

    return model
