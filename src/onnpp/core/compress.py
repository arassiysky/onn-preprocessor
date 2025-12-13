from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class CompressionReport:
    method: str
    original_len: int
    compressed_len: int
    kept_indices: List[int]


@torch.no_grad()
def rle_compress_input_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    min_run: int = 2,
    keep: str = "first",
) -> Tuple[torch.Tensor, torch.Tensor, CompressionReport]:
    """
    Deterministic run-length compression for token IDs.

    input_ids: (B,S)
    attention_mask: (B,S) or None (if None, assumed all ones)

    Returns:
      input_ids_c: (B,S')
      attention_mask_c: (B,S')
      report: CompressionReport (indices kept from the original sequence)
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be (B,S). Got {tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")

    B, S = input_ids.shape
    device = input_ids.device

    # For v0.M1, we support B=1 cleanly (benchmarks + quickstart). Extend later.
    if B != 1:
        raise NotImplementedError("v0.M1 rle compression currently supports batch_size=1 for simplicity.")

    ids = input_ids[0]
    mask = attention_mask[0]

    # Only consider tokens where mask==1
    valid_len = int(mask.sum().item())
    ids = ids[:valid_len]

    kept: List[int] = []
    i = 0
    while i < valid_len:
        j = i + 1
        while j < valid_len and ids[j].item() == ids[i].item():
            j += 1
        run_len = j - i

        if run_len >= min_run:
            kept_idx = i if keep == "first" else (j - 1)
            kept.append(kept_idx)
        else:
            kept.extend(list(range(i, j)))

        i = j

    kept_t = torch.tensor(kept, dtype=torch.long, device=device)
    ids_c = ids.index_select(0, kept_t).view(1, -1)
    mask_c = torch.ones((1, ids_c.shape[1]), dtype=torch.int64, device=device)

    report = CompressionReport(
        method=f"rle(min_run={min_run}, keep={keep})",
        original_len=valid_len,
        compressed_len=int(ids_c.shape[1]),
        kept_indices=kept,
    )
    return ids_c, mask_c, report

@torch.no_grad()
def onn_gated_compress_input_ids(
    input_ids: torch.Tensor,
    features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    threshold: float | None,
    quantile: float | None,
    min_run: int,
):
    """
    Operator-gated compression based on ONN feature magnitude.
    Compression triggers where score < threshold, for spans of length >= min_run.

    If quantile is not None, threshold is computed as quantile(scores).
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    if input_ids.ndim != 2 or features.ndim != 3:
        raise ValueError(f"Expected input_ids (B,S) and features (B,S,K), got {tuple(input_ids.shape)} and {tuple(features.shape)}")

    B, S = input_ids.shape
    if B != 1:
        raise NotImplementedError("onn_gated compression currently supports batch_size=1 (v0.M).")

    ids = input_ids[0]
    F = features[0]
    mask = attention_mask[0]

    valid_len = int(mask.sum().item())
    ids = ids[:valid_len]
    F = F[:valid_len]

    # scalar "information" score per token
    scores = F.abs().mean(dim=-1)  # (S,)

    # derive threshold
    used_threshold: float
    used_quantile: float | None = None
    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0,1)")
        used_quantile = float(quantile)
        used_threshold = float(torch.quantile(scores, torch.tensor(used_quantile, device=scores.device)))
    else:
        if threshold is None:
            raise ValueError("Either threshold or quantile must be provided for onn_gated compression.")
        used_threshold = float(threshold)

    kept = []
    i = 0
    while i < valid_len:
        if float(scores[i]) >= used_threshold:
            kept.append(i)
            i += 1
            continue

        j = i + 1
        while j < valid_len and float(scores[j]) < used_threshold:
            j += 1

        run_len = j - i
        if run_len >= min_run:
            kept.append(i)  # keep first
        else:
            kept.extend(range(i, j))
        i = j

    kept_t = torch.tensor(kept, dtype=torch.long, device=input_ids.device)
    ids_c = ids.index_select(0, kept_t).view(1, -1)
    mask_c = torch.ones_like(ids_c, dtype=torch.int64)

    report = {
        "method": "onn_gated",
        "original_len": valid_len,
        "compressed_len": int(ids_c.shape[1]),
        "threshold": used_threshold,
        "quantile": used_quantile,
        "min_run": int(min_run),
    }
    return ids_c, mask_c, report

