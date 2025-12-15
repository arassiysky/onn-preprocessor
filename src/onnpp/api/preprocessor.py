from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from onnpp.core.config import ONNConfig
from onnpp.core.featurizer import SimpleTokenFeaturizer
from onnpp.core.projector import ConcatProjector, ResidualDeterministicProjector
from onnpp.core.compress import rle_compress_input_ids, onn_gated_compress_input_ids


EmbedFn = Callable[[torch.Tensor], torch.Tensor]


# ============================================================
# L2: Adaptive policy
# ============================================================

class L2Policy:
    """
    Adaptive L2 policy based on feature scores.

    Key refinement:
      - Different aggressiveness for greedy vs sampling
      - Greedy can compress more (higher quantile, higher max_drop_frac)
      - Sampling stays conservative (clamped quantile, lower max_drop_frac)
    """

    def __init__(
        self,
        *,
        # enable threshold
        min_len_enable: int = 256,

        # Greedy knobs
        greedy_min_saved_frac: float = 0.04,
        greedy_max_drop_frac: float = 0.40,
        greedy_quantile_cap: float = 0.40,

        # Sampling knobs
        sampling_min_saved_frac: float = 0.06,
        sampling_max_drop_frac: float = 0.30,
        sampling_quantile_cap: float = 0.25,

        # min_run
        min_run_code: int = 2,
        min_run_text: int = 2,
    ):
        self.min_len_enable = int(min_len_enable)

        self.greedy_min_saved_frac = float(greedy_min_saved_frac)
        self.greedy_max_drop_frac = float(greedy_max_drop_frac)
        self.greedy_quantile_cap = float(greedy_quantile_cap)

        self.sampling_min_saved_frac = float(sampling_min_saved_frac)
        self.sampling_max_drop_frac = float(sampling_max_drop_frac)
        self.sampling_quantile_cap = float(sampling_quantile_cap)

        self.min_run_code = int(min_run_code)
        self.min_run_text = int(min_run_text)

    def _base_quantile(self, seq_len: int, task_mode: str, *, is_sampling: bool) -> float:
        """
        Greedy schedule is a bit more aggressive than sampling schedule.
        """
        if is_sampling:
            # conservative schedule
            if seq_len < 256:
                q = 0.15
            elif seq_len < 512:
                q = 0.22
            elif seq_len < 1024:
                q = 0.25
            else:
                q = 0.27
            cap = self.sampling_quantile_cap
        else:
            # greedy: push harder at long contexts
            if seq_len < 256:
                q = 0.18
            elif seq_len < 512:
                q = 0.25
            elif seq_len < 1024:
                q = 0.30
            else:
                q = 0.34
            cap = self.greedy_quantile_cap

        if task_mode == "code":
            q += 0.03

        return float(max(0.05, min(cap, q)))

    def _predict_kept(
        self,
        scores: torch.Tensor,
        threshold: float,
        min_run: int,
    ) -> int:
        S = int(scores.numel())
        kept = 0
        i = 0
        while i < S:
            if float(scores[i]) >= threshold:
                kept += 1
                i += 1
                continue
            j = i + 1
            while j < S and float(scores[j]) < threshold:
                j += 1
            run_len = j - i
            kept += 1 if run_len >= min_run else run_len
            i = j
        return kept

    def decide_from_scores(
    self,
    *,
    seq_len: int,
    task_mode: str,
    is_sampling: bool,
    scores: torch.Tensor,
) -> Dict[str, Any]:
    # =========================================================
    # 0) Hard gates (must NOT be nested under min_len_enable)
    # =========================================================

    # HARD disable ONN for short sequences (avoid overhead)
    if seq_len < 300:
        return {
            "enabled": False,
            "quantile": None,
            "min_run": self.min_run_code if task_mode == "code" else self.min_run_text,
            "note": "disabled: short(<300)",
        }

    # HARD disable ONN compression at very long contexts (empirical: avoids decode regressions)
    # Applies to BOTH greedy and sampling (based on your latest benchmark evidence).
    if seq_len >= 900:
        return {
            "enabled": False,
            "quantile": None,
            "min_run": self.min_run_code if task_mode == "code" else self.min_run_text,
            "note": "disabled: long(>=900)",
        }

    # (Optional) Greedy-only extra guard band (keep if you want stricter greedy control)
    # If you keep the >=900 global gate, you can delete this block entirely.
    # if (not is_sampling) and (seq_len >= 800):
    #     return {
    #         "enabled": False,
    #         "quantile": None,
    #         "min_run": self.min_run_code if task_mode == "code" else self.min_run_text,
    #         "note": "disabled: greedy-long(>=800)",
    #     }

    # =========================================================
    # 1) Normal policy logic
    # =========================================================

    min_run = self.min_run_code if task_mode == "code" else self.min_run_text

    # choose regime-specific thresholds
    if is_sampling:
        min_saved = self.sampling_min_saved_frac
        max_drop = self.sampling_max_drop_frac
    else:
        min_saved = self.greedy_min_saved_frac
        max_drop = self.greedy_max_drop_frac

    # Extra strictness in the borderline regime (helps avoid 256 regressions)
    if seq_len < 384:
        min_saved = max(min_saved, 0.08)

    q = self._base_quantile(seq_len, task_mode, is_sampling=is_sampling)

    thr = float(torch.quantile(scores, torch.tensor(q, device=scores.device)))
    kept = self._predict_kept(scores, thr, min_run)
    saved_frac = 1.0 - kept / float(seq_len)

    # disable if too little predicted saving (avoid overhead regression)
    if saved_frac < min_saved:
        return {
            "enabled": False,
            "quantile": None,
            "min_run": min_run,
            "note": f"disabled: saved={saved_frac:.3f} < min_saved={min_saved:.3f}",
        }

    # if too aggressive, back off until within max_drop
    while saved_frac > max_drop and q > 0.05:
        q -= 0.03
        thr = float(torch.quantile(scores, torch.tensor(q, device=scores.device)))
        kept = self._predict_kept(scores, thr, min_run)
        saved_frac = 1.0 - kept / float(seq_len)

    mode = "sampling" if is_sampling else "greedy"
    return {
        "enabled": True,
        "quantile": float(q),
        "min_run": int(min_run),
        "note": f"enabled({mode}): saved={saved_frac:.3f} q={q:.2f} max_drop={max_drop:.2f}",
    }


def _l2_policy_from_runtime(model: Any | None, cfg: ONNConfig) -> L2Policy:
    if model is not None:
        pol = getattr(model, "onn_l2_policy", None)
        if isinstance(pol, L2Policy):
            return pol

    def g(name: str, default):
        if model is not None and hasattr(model, name):
            return getattr(model, name)
        return getattr(cfg, name, default)

    return L2Policy(
        min_len_enable=int(g("onn_l2_min_len_enable", 256)),

        # Greedy knobs (more aggressive)
        greedy_min_saved_frac=float(g("onn_l2_greedy_min_saved_frac", 0.04)),
        greedy_max_drop_frac=float(g("onn_l2_greedy_max_drop_frac", 0.40)),
        greedy_quantile_cap=float(g("onn_l2_greedy_quantile_cap", 0.40)),

        # Sampling knobs (conservative)
        sampling_min_saved_frac=float(g("onn_l2_sampling_min_saved_frac", 0.06)),
        sampling_max_drop_frac=float(g("onn_l2_sampling_max_drop_frac", 0.30)),
        sampling_quantile_cap=float(g("onn_l2_sampling_quantile_cap", 0.25)),

        # Min-run
        min_run_code=int(g("onn_l2_min_run_code", 2)),
        min_run_text=int(g("onn_l2_min_run_text", 2)),
    )


# ============================================================
# L4: positional mask
# ============================================================

def _l4_weights(seq_len: int, device: torch.device, *, alpha: float) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    mask = ((idx & (idx >> 1)) == 0).float()
    return 1.0 + alpha * mask


# ============================================================
# ONNPreprocessor
# ============================================================

class ONNPreprocessor(torch.nn.Module):
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
            if config.projector_kind == "concat":
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
        E = embed_fn(input_ids)
        F, feat_rep = self.featurizer(input_ids)
        E_aug, proj_rep = self.projector(E, F)

        return E_aug, {
            "onnpp_version": self.config.version,
            "config": asdict(self.config),
            "shapes": {
                "input_ids": tuple(input_ids.shape),
                "E": tuple(E.shape),
                "F": tuple(F.shape),
                "E_aug": tuple(E_aug.shape),
            },
            "featurizer": feat_rep,
            "projector": proj_rep,
        }

    @torch.no_grad()
    def compress_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        model: Any | None = None,
        task_mode: str = "text",
        is_sampling: bool | None = None,
    ):
        cfg = self.config

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        if attention_mask.dtype != torch.int64:
            attention_mask = attention_mask.to(torch.int64)

        valid_len = int(attention_mask[0].sum().item())
        kind = getattr(cfg, "compression", "none")

        if kind == "none":
            return input_ids, attention_mask, {
                "compression": {"method": "none", "original_len": valid_len, "compressed_len": valid_len}
            }

        if is_sampling is None and model is not None:
            is_sampling = bool(getattr(model, "onn_is_sampling", False))
        if is_sampling is None:
            is_sampling = False

        if kind == "rle":
            ids_c, mask_c, rep = rle_compress_input_ids(
                input_ids,
                attention_mask,
                min_run=int(getattr(cfg, "rle_min_run", 2)),
                keep=str(getattr(cfg, "rle_keep", "first")),
            )
            return ids_c, mask_c, {"compression": rep.__dict__}

        if kind == "onn_gated":
            F, _ = self.featurizer(input_ids)
            scores = F[0].abs().mean(dim=-1)

            pol = _l2_policy_from_runtime(model, cfg)
            dec = pol.decide_from_scores(
                seq_len=int(scores.numel()),
                task_mode=task_mode,
                is_sampling=bool(is_sampling),
                scores=scores,
            )

            if not dec["enabled"]:
                return input_ids, attention_mask, {
                    "compression": {
                        "method": "none(L2)",
                        "original_len": valid_len,
                        "compressed_len": valid_len,
                        "note": dec["note"],
                    }
                }

            eff_q = dec["quantile"]
            eff_min_run = dec["min_run"]

            if bool(getattr(cfg, "l4_enabled", True)):
                alpha = float(getattr(cfg, "l4_alpha", 0.5))
                w = _l4_weights(scores.numel(), scores.device, alpha=alpha).to(F.dtype)
                F = F * w.view(1, -1, 1)

            ids_c, mask_c, rep = onn_gated_compress_input_ids(
                input_ids,
                F,
                attention_mask,
                threshold=float(getattr(cfg, "onn_gate_threshold", 0.1)),
                quantile=eff_q,
                min_run=int(eff_min_run),
            )

            rep = dict(rep)
            rep["l2_note"] = dec["note"]
            return ids_c, mask_c, {"compression": rep}

        raise ValueError(f"Unknown compression kind: {kind}")


