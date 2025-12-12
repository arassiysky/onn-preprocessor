from __future__ import annotations

from typing import Dict, Tuple

import torch


class ConcatProjector(torch.nn.Module):
    """
    Projects concatenated [E | F] back to d_model.

    E: (B,S,D), F: (B,S,K) -> concat: (B,S,D+K) -> proj -> (B,S,D)
    """

    def __init__(self, d_model: int, feature_dim: int, init: str = "identity_like"):
        super().__init__()
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)
        self.in_dim = self.d_model + self.feature_dim

        self.proj = torch.nn.Linear(self.in_dim, self.d_model, bias=True)
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        if init == "xavier":
            torch.nn.init.xavier_uniform_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)
            return

        if init == "identity_like":
            # Make output start close to original embedding E (pass-through),
            # while initially ignoring F. This stabilizes early experiments.
            torch.nn.init.zeros_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)

            # Set the first D columns to identity
            D = self.d_model
            with torch.no_grad():
                self.proj.weight[:, :D] = torch.eye(D, dtype=self.proj.weight.dtype, device=self.proj.weight.device)
            return

        raise ValueError(f"Unknown init '{init}'. Use 'xavier' or 'identity_like'.")

    def forward(self, E: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if E.ndim != 3:
            raise ValueError(f"E must be (B,S,D). Got {tuple(E.shape)}")
        if F.ndim != 3:
            raise ValueError(f"F must be (B,S,K). Got {tuple(F.shape)}")

        if E.shape[:2] != F.shape[:2]:
            raise ValueError(f"Batch/seq mismatch: E {tuple(E.shape)} vs F {tuple(F.shape)}")
        if E.shape[-1] != self.d_model:
            raise ValueError(f"E last dim must be d_model={self.d_model}. Got {E.shape[-1]}")
        if F.shape[-1] != self.feature_dim:
            raise ValueError(f"F last dim must be feature_dim={self.feature_dim}. Got {F.shape[-1]}")

        X = torch.cat([E, F], dim=-1)  # (B,S,D+K)
        out = self.proj(X)             # (B,S,D)

        report = {
            "projector": "ConcatProjector(v0.L)",
            "in_dim": self.in_dim,
            "out_dim": self.d_model,
        }
        return out, report


class ResidualDeterministicProjector(torch.nn.Module):
    """
    Deterministic, non-trainable residual injection:

        E_aug = E + epsilon * (F @ W)

    where W is a fixed matrix (K x D) generated once from a seed and stored as a buffer.
    """

    def __init__(self, d_model: int, feature_dim: int, epsilon: float = 0.01, seed: int = 42):
        super().__init__()
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)
        self.seed = int(seed)

        # Create a fixed weight W of shape (K, D) deterministically
        g = torch.Generator()
        g.manual_seed(self.seed)

        W = torch.empty(self.feature_dim, self.d_model, dtype=torch.float32)
        # Deterministic Xavier-like init
        torch.nn.init.xavier_uniform_(W, generator=g)

        # Register as buffer => saved with model, moved with .to(device), NOT trainable
        self.register_buffer("W", W)

    def forward(self, E: torch.Tensor, F: torch.Tensor):
        if E.ndim != 3:
            raise ValueError(f"E must be (B,S,D). Got {tuple(E.shape)}")
        if F.ndim != 3:
            raise ValueError(f"F must be (B,S,K). Got {tuple(F.shape)}")

        if E.shape[:2] != F.shape[:2]:
            raise ValueError(f"Batch/seq mismatch: E {tuple(E.shape)} vs F {tuple(F.shape)}")
        if E.shape[-1] != self.d_model:
            raise ValueError(f"E last dim must be d_model={self.d_model}. Got {E.shape[-1]}")
        if F.shape[-1] != self.feature_dim:
            raise ValueError(f"F last dim must be feature_dim={self.feature_dim}. Got {F.shape[-1]}")

        # (B,S,K) @ (K,D) -> (B,S,D)
        delta = torch.matmul(F.to(self.W.dtype), self.W)

        out = E + (self.epsilon * delta).to(E.dtype)

        report = {
            "projector": "ResidualDeterministicProjector(v0.L)",
            "epsilon": self.epsilon,
            "seed": self.seed,
            "feature_dim": self.feature_dim,
            "out_dim": self.d_model,
        }
        return out, report
