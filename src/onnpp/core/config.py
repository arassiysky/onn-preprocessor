from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ProjectorInit = Literal["xavier", "identity_like"]


@dataclass(frozen=True)
class ONNConfig:
    """
    v0.L config: embedding augmentation only.
    ONN produces per-token features F (B,S,K), concatenates to embeddings E (B,S,D),
    then projects back to (B,S,D).
    """

    # LLM embedding dimension (Llama d_model)
    d_model: int

    # Feature dimension produced by featurizer
    feature_dim: int = 8

    # Deterministic features only for v0.L
    deterministic: bool = True

    # Projector init strategy
    projector_init: ProjectorInit = "identity_like"

    # Optional: bound features to [-clip, +clip] for stability (None disables)
    feature_clip: Optional[float] = 5.0

    # Version label for reporting
    version: str = "0.0.L"

    @staticmethod
    def v0L(d_model: int, feature_dim: int = 8) -> "ONNConfig":
        return ONNConfig(d_model=d_model, feature_dim=feature_dim)