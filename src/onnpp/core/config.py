from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ProjectorKind = Literal["concat", "residual_det"]
ProjectorInit = Literal["xavier", "identity_like"]
CompressionKind = Literal["none", "rle", "onn_gated"]
RLEKeep = Literal["first", "last"]


@dataclass(frozen=True)
class ONNConfig:
    # --- REQUIRED (must be first) ---
    d_model: int  # LLM embedding dimension (Llama hidden size)

    # --- Option 2 (augment) ---
    feature_dim: int = 8
    deterministic: bool = True

    projector_kind: ProjectorKind = "concat"
    projector_init: ProjectorInit = "identity_like"

    # Residual deterministic projector params
    epsilon: float = 0.001
    seed: int = 42

    # Feature stability
    feature_clip: Optional[float] = 5.0

    # --- Option 1 (compress) ---
    compression: CompressionKind = "none"
    rle_min_run: int = 2
    rle_keep: RLEKeep = "first"

    onn_gate_threshold: float = 0.1   # τ
    onn_gate_min_run: int = 2
    onn_gate_quantile: float | None = 0.25  # if set, overrides threshold (e.g. bottom 25% = "low-info")

    # Metadata
    version: str = "0.0.L"

    @staticmethod
    def v0L(d_model: int, feature_dim: int = 8) -> "ONNConfig":
        return ONNConfig(d_model=d_model, feature_dim=feature_dim)