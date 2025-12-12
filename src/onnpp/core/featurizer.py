from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class FeatureReport:
    feature_dim: int
    notes: str


class SimpleTokenFeaturizer(torch.nn.Module):
    """
    v0.L minimal deterministic featurizer.
    Produces cheap per-token features from input_ids only.

    Output: F in shape (B, S, K)
    """

    def __init__(self, feature_dim: int = 8, clip: float | None = 5.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.clip = clip

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        input_ids: (B, S) int64
        returns:
          F: (B, S, K) float32
          report: dict
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B,S). Got {tuple(input_ids.shape)}")

        B, S = input_ids.shape
        device = input_ids.device

        # Base signals (all float32)
        ids_f = input_ids.to(torch.float32)

        # 1) parity of token id (LSB)
        parity = (input_ids & 1).to(torch.float32)

        # 2) local equality with next token (shifted)
        next_ids = torch.roll(input_ids, shifts=-1, dims=1)
        eq_next = (input_ids == next_ids).to(torch.float32)

        # 3) local delta magnitude (bounded)
        delta = (next_ids.to(torch.int64) - input_ids.to(torch.int64)).abs().to(torch.float32)

        # 4) normalized position in sequence [0,1]
        pos = torch.linspace(0.0, 1.0, steps=S, device=device, dtype=torch.float32).view(1, S).expand(B, S)

        # Assemble a base bank of features (B,S,4)
        base = torch.stack([parity, eq_next, delta, pos], dim=-1)

        # Expand / pad to requested K
        K = self.feature_dim
        if K <= base.shape[-1]:
            F = base[..., :K].contiguous()
        else:
            # repeat base and crop
            reps = (K + base.shape[-1] - 1) // base.shape[-1]
            F = base.repeat(1, 1, reps)[..., :K].contiguous()

        # Optional clipping
        if self.clip is not None:
            F = torch.clamp(F, -float(self.clip), float(self.clip))

        report = {
            "feature_dim": K,
            "featurizer": "SimpleTokenFeaturizer(v0.L)",
            "notes": "Deterministic token-id features: parity, eq_next, delta, position (tiled/padded).",
        }
        return F, report
