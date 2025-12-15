# src/onnpp/api/preprocessor.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from onnpp.core.config import ONNConfig
from onnpp.core.featurizer import SimpleTokenFeaturizer
from onnpp.core.projector import ConcatProjector, ResidualDeterministicProjector

# compression primitives
from onnpp.core.compress import rle_compress_input_ids, onn_gated_compress_input_ids


EmbedFn = Callable[[torch.Tensor], torch.Tensor]


# ============================================================
# L4: lightweight deterministic score modulation ("mask")
# ============================================================

@torch.no_grad()
def l4_modulate_scores(
    input_ids: torch.Tensor,  # (B,S)
    scores: torch.Tensor,     # (S,)  (valid_len already applied)
    *,
    strength: float = 0.15,
) -> torch.Tensor:
    """
    Deterministic modulation of score stream to introduce structured "phase"
    without changing model weights. This is intentionally cheap.

    strength in [0, ~0.3] recommended.
    """
    if strength <= 0.0:
        return scores

    # positions 0..S-1
    S = int(scores.numel())
    device = scores.device
    pos = torch.arange(S, device=device, dtype=torch.long)

    # simple Sierpinski-like boolean via bitwise AND
    # 1 where (pos & (pos >> 1)) == 0 else 0
    sier = ((pos & (pos >> 1)) == 0).to(torch.float32)  # (S,)

    # token hash (cheap deterministic)
    # use low bits of token id
    ids = input_ids[0, :S].to(torch.long)
    h = (ids ^ (ids >> 3) ^ (ids << 1)) & 0xF
    h = h.to(torch.float32) / 15.0  # [0,1]

    # combine into multiplier in [1-strength, 1+strength]
    phase = (2.0 * (0.6 * sier + 0.4 * h) - 1.0)  # roughly [-1,1]
    mult = 1.0 + float(strength) * phase
    return scores * mult


# ============================================================
# L2: adaptive policy (greedy vs sampling aware)
# ============================================================

class L2Policy:
    """
    Adaptive L2 policy that decides whether to enable compression, and if so,
    which quantile + min_run to use, based on scores.

    Design goals:
      - hard-disable for short sequences (avoid overhead)
      - hard-disable for very long sequences (avoid KV/cache regressions)
      - greedy can be more aggressive than sampling (but still bounded)
    """

    def __init__(
        self,
        *,
        # hard gates
        min_len_enable: int = 300,
        max_len_enable: int = 900,  # disable at/above this length (both greedy+sampling)

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
        self.max_len_enable = int(max_len_enable)

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
        Heuristic schedule. Code tends to tolerate a bit more compression.
        """
        if is_sampling:
            # conservative
            if seq_len < 384:
                q = 0.15
            elif seq_len < 768:
                q = 0.22
            else:
                q = 0.25
            cap = self.sampling_quantile_cap
        else:
            # greedier
            if seq_len < 384:
                q = 0.20
            elif seq_len < 768:
                q = 0.28
            else:
                q = 0.34
            cap = self.greedy_quantile_cap

        if task_mode == "code":
            q += 0.03

        q = max(0.05, min(cap, q))
        return float(q)

    @staticmethod
    def _predict_kept(scores: torch.Tensor, threshold: float, min_run: int) -> int:
        """
        Predict kept tokens for onn_gated rule:
          - keep tokens with score >= threshold
          - for spans score < threshold with length >= min_run, keep first
          - else keep all
        """
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
        # -------------------------
        # Hard gates
        # -------------------------
        if seq_len < self.min_len_enable:
            return {
                "enabled": False,
                "quantile": None,
                "min_run": self.min_run_code if task_mode == "code" else self.min_run_text,
                "note": f"disabled: short(<{self.min_len_enable})",
            }

        if seq_len >= self.max_len_enable:
            return {
                "enabled": False,
                "quantile": None,
                "min_run": self.min_run_code if task_mode == "code" else self.min_run_text,
                "note": f"disabled: long(>={self.max_len_enable})",
            }

        min_run = self.min_run_code if task_mode == "code" else self.min_run_text

        # -------------------------
        # Regime knobs
        # -------------------------
        if is_sampling:
            min_saved = self.sampling_min_saved_frac
            max_drop = self.sampling_max_drop_frac
        else:
            min_saved = self.greedy_min_saved_frac
            max_drop = self.greedy_max_drop_frac

        # extra strictness in borderline region (helps avoid 256 regressions)
        if seq_len < 384:
            min_saved = max(min_saved, 0.08)

        q = self._base_quantile(seq_len, task_mode, is_sampling=is_sampling)

        thr = float(torch.quantile(scores, torch.tensor(q, device=scores.device)))
        kept = self._predict_kept(scores, thr, min_run)
        saved_frac = 1.0 - kept / float(seq_len)

        if saved_frac < min_saved:
            return {
                "enabled": False,
                "quantile": None,
                "min_run": min_run,
                "note": f"disabled: saved={saved_frac:.3f} < min_saved={min_saved:.3f}",
            }

        # back off if too aggressive
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
    """
    Builds L2Policy from:
      1) model.onn_l2_policy if provided
      2) model attributes (runtime knobs)
      3) cfg attributes (fallback)
    Backwards compatible with older knob names:
      - onn_l2_min_saved_frac / onn_l2_max_drop_frac / onn_l2_sampling_quantile_cap
    """
    if model is not None:
        pol = getattr(model, "onn_l2_policy", None)
        if isinstance(pol, L2Policy):
            return pol

    def g(name: str, default):
        if model is not None and hasattr(model, name):
            return getattr(model, name)
        return getattr(cfg, name, default)

    # Back-compat names
    bc_min_saved = float(g("onn_l2_min_saved_frac", 0.06))
    bc_max_drop = float(g("onn_l2_max_drop_frac", 0.30))
    bc_samp_cap = float(g("onn_l2_sampling_quantile_cap", 0.25))

    return L2Policy(
        min_len_enable=int(g("onn_l2_min_len_enable", 300)),
        max_len_enable=int(g("onn_l2_max_len_enable", 900)),

        greedy_min_saved_frac=float(g("onn_l2_greedy_min_saved_frac", bc_min_saved)),
        greedy_max_drop_frac=float(g("onn_l2_greedy_max_drop_frac", max(0.40, bc_max_drop))),
        greedy_quantile_cap=float(g("onn_l2_greedy_quantile_cap", 0.40)),

        sampling_min_saved_frac=float(g("onn_l2_sampling_min_saved_frac", bc_min_saved)),
        sampling_max_drop_frac=float(g("onn_l2_sampling_max_drop_frac", bc_max_drop)),
        sampling_quantile_cap=float(g("onn_l2_sampling_quantile_cap", bc_samp_cap)),

        min_run_code=int(g("onn_l2_min_run_code", 2)),
        min_run_text=int(g("onn_l2_min_run_text", 2)),
    )


# ============================================================
# ONN Preprocessor
# ============================================================

class ONNPreprocessor(torch.nn.Module):
    """
    ONN preprocessor:
      - Option 2: embedding augmentation (L1/L3)
      - Option 1: compression (RLE or ONN-gated)
      - L2: adaptive policy for ONN-gated compression
      - L4: optional deterministic score modulation (mask)
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
            elif getattr(config, "projector_kind", "concat") == "residual_det":
                self.projector = ResidualDeterministicProjector(
                    d_model=config.d_model,
                    feature_dim=config.feature_dim,
                    epsilon=config.epsilon,
                    seed=config.seed,
                )
            else:
                raise ValueError(f"Unknown projector_kind: {getattr(config, 'projector_kind', None)}")

    @torch.no_grad()
    def augment_embeddings(
        self,
        input_ids: torch.Tensor,
        embed_fn: EmbedFn,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        input_ids: (B,S) int64
        embed_fn: maps input_ids -> embeddings (B,S,D)
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B,S). Got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise ValueError(f"input_ids must be int tensor. Got dtype={input_ids.dtype}")

        # base embeddings
        E = embed_fn(input_ids)
        if E.ndim != 3:
            raise ValueError(f"embed_fn must return (B,S,D). Got {tuple(E.shape)}")
        if E.shape[0] != input_ids.shape[0] or E.shape[1] != input_ids.shape[1]:
            raise ValueError(f"embed_fn output must match (B,S). Got E={tuple(E.shape)}, ids={tuple(input_ids.shape)}")
        if int(E.shape[-1]) != int(self.config.d_model):
            raise ValueError(f"embed dim mismatch: expected D={self.config.d_model}, got {E.shape[-1]}")

        # features
        F, feat_report = self.featurizer(input_ids)

        # project
        E_aug, proj_report = self.projector(E, F)

        report: Dict[str, Any] = {
            "onnpp_version": getattr(self.config, "version", "unknown"),
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

    @torch.no_grad()
    def compress_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        model: Any | None = None,
        task_mode: str | None = None,
    ):
        """
        Option 1: compression. Supports:
          - none
          - rle
          - onn_gated (with adaptive L2 + optional L4)

        Returns: (input_ids_c, attention_mask_c, report_dict)

        Note: model is optional. If provided, L2 can read runtime flags:
          - model.onn_task_mode (code/text)
          - model.onn_is_sampling (bool)  [set by benchmark before generate]
          - model.onn_l2_enabled (bool)
          - model.onn_l4_enabled (bool)
        """
        cfg = self.config

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

        if attention_mask.dtype != torch.int64:
            attention_mask = attention_mask.to(torch.int64)

        comp_kind = getattr(cfg, "compression", "none")

        if comp_kind == "none":
            valid_len = int(attention_mask.sum().item())
            return input_ids, attention_mask, {
                "compression": {"method": "none", "original_len": valid_len, "compressed_len": valid_len}
            }

        if comp_kind == "rle":
            ids_c, mask_c, rep = rle_compress_input_ids(
                input_ids,
                attention_mask,
                min_run=getattr(cfg, "rle_min_run", 2),
                keep=getattr(cfg, "rle_keep", "first"),
            )
            return ids_c, mask_c, {"compression": rep.__dict__}

        if comp_kind == "onn_gated":
            # Determine task mode / sampling flag
            tm = task_mode
            if tm is None:
                tm = getattr(model, "onn_task_mode", None) if model is not None else None
            if tm is None:
                tm = "text"

            is_sampling = bool(getattr(model, "onn_is_sampling", False)) if model is not None else False

            # L2 & L4 toggles
            l2_enabled = bool(getattr(model, "onn_l2_enabled", True)) if model is not None else True
            l4_enabled = bool(getattr(model, "onn_l4_enabled", True)) if model is not None else False

            # Features (cheap O(S))
            F, _ = self.featurizer(input_ids)

            # scalar scores per token
            scores = F[0].abs().mean(dim=-1)  # (S,)

            # apply valid_len from attention_mask
            valid_len = int(attention_mask[0].sum().item())
            scores = scores[:valid_len]

            # L4 modulation (optional)
            l4_strength = float(getattr(cfg, "l4_strength", 0.15))
            if l4_enabled:
                scores = l4_modulate_scores(input_ids, scores, strength=l4_strength)

            # Decide quantile/min_run
            used_quantile = getattr(cfg, "onn_gate_quantile", None)
            used_threshold = getattr(cfg, "onn_gate_threshold", 0.1)
            used_min_run = int(getattr(cfg, "onn_gate_min_run", 2))
            l2_note = ""

            if l2_enabled:
                pol = _l2_policy_from_runtime(model, cfg) if model is not None else _l2_policy_from_runtime(None, cfg)
                dec = pol.decide_from_scores(
                    seq_len=valid_len,
                    task_mode=str(tm),
                    is_sampling=is_sampling,
                    scores=scores,
                )
                l2_note = str(dec.get("note", ""))

                if not bool(dec.get("enabled", False)):
                    # disable compression (pass-through)
                    return input_ids, attention_mask, {
                        "compression": {
                            "method": "onn_gated",
                            "original_len": valid_len,
                            "compressed_len": valid_len,
                            "enabled": False,
                            "l2_note": l2_note,
                            "l4_enabled": bool(l4_enabled),
                        }
                    }

                used_quantile = dec.get("quantile", used_quantile)
                used_min_run = int(dec.get("min_run", used_min_run))

            # Perform gating compression (uses features, but we pass the original F)
            ids_c, mask_c, rep = onn_gated_compress_input_ids(
                input_ids,
                F,
                attention_mask,
                threshold=float(used_threshold) if used_quantile is None else None,
                quantile=float(used_quantile) if used_quantile is not None else None,
                min_run=int(used_min_run),
            )

            # enrich report
            rep["enabled"] = True
            rep["l2_note"] = l2_note
            rep["l4_enabled"] = bool(l4_enabled)

            return ids_c, mask_c, {"compression": rep}

        raise ValueError(f"Unknown compression kind: {comp_kind}")
