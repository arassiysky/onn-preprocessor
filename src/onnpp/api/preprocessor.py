from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from onnpp.core.config import ONNConfig
from onnpp.core.featurizer import SimpleTokenFeaturizer
from onnpp.core.projector import ConcatProjector, ResidualDeterministicProjector


EmbedFn = Callable[[torch.Tensor], torch.Tensor]


class ONNPreprocessor(torch.nn.Module):
    """
    v0.L: Embedding augmentation pre-processor.

    Usage:
        onn = ONNPreprocessor(ONNConfig.v0L(d_model=4096, feature_dim=8))
        E_aug, report = onn.augment_embeddings(input_ids, embed_fn=llama_embed)
    """

    def __init__(
        self,
        config: ONNConfig,
        featurizer: Optional[torch.nn.Module] = None,
        projector: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        self.featurizer = featurizer or SimpleTokenFeaturizer(
            feature_dim=config.feature_dim,
            clip=config.feature_clip,
        )

        if projector is not None:
            self.projector = projector
        else:
            if getattr(config, "projector_kind", "concat") == "concat":
                self.projector = ConcatProjector(
                    d_model=config.d_model,
                    feature_dim=config.feature_dim,
                    init=config.projector_init,
                )
            elif config.projector_kind == "residual_det":
                self.projector = ResidualDeterministicProjector(
                    d_model=config.d_model,
                    feature_dim=config.feature_dim,
                    epsilon=config.epsilon,
                    seed=config.seed,
                )
            else:
                raise ValueError(f"Unknown projector_kind: {config.projector_kind}")

    @torch.no_grad()
    def augment_embeddings(
        self,
        input_ids: torch.Tensor,
        embed_fn: EmbedFn,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        input_ids: (B,S) int64
        embed_fn: function mapping input_ids -> embeddings E (B,S,D)
        returns:
          E_aug: (B,S,D)
          report: dict of metadata + shapes
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B,S). Got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"input_ids must be int tensor. Got dtype={input_ids.dtype}")

        # 1) base embeddings from user model
        E = embed_fn(input_ids)
        if E.ndim != 3:
            raise ValueError(f"embed_fn must return (B,S,D). Got {tuple(E.shape)}")
        if E.shape[0] != input_ids.shape[0] or E.shape[1] != input_ids.shape[1]:
            raise ValueError(f"embed_fn output must match (B,S). Got E={tuple(E.shape)}, ids={tuple(input_ids.shape)}")
        if E.shape[-1] != self.config.d_model:
            raise ValueError(f"embed dim mismatch: expected D={self.config.d_model}, got {E.shape[-1]}")

        # 2) ONN features
        F, feat_report = self.featurizer(input_ids)

        # 3) fuse and project back to D
        E_aug, proj_report = self.projector(E, F)

        report: Dict[str, Any] = {
            "onnpp_version": self.config.version,
            "config": asdict(self.config),
            "shapes": {
                "input_ids": tuple(input_ids.shape),
                "E": tuple(E.shape),
                "F": tuple(F.shape),
                "E_aug": tuple(E_aug.shape),
            },
            "featurizer": feat_report,
            "projector": proj_report,
        }
        return E_aug, report
