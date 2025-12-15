from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class L2Decision:
    """
    L2 decision outputs effective compression settings for this forward pass.

    - enabled: whether compression should run
    - quantile: effective quantile (if using onn_gated)
    - min_run: effective min_run
    - note: short reason string for debugging
    """
    enabled: bool
    quantile: Optional[float]
    min_run: int
    note: str


@dataclass
class L2Policy:
    """
    L2 = control plane for operator application.

    This policy is intentionally simple and deterministic:
    - Disable compression for short sequences (overhead > savings).
    - Use safer/less aggressive quantiles for sampling (stability).
    - Allow per-task-mode defaults (code vs text).
    """

    # Below this length, compression usually loses (overhead dominates)
    min_len_enable: int = 128

    # Conservative quantile when sampling (stability)
    sampling_quantile_cap: float = 0.25

    # If caller didn't set quantile, use these defaults by mode
    default_quantile_code: float = 0.25
    default_quantile_text: float = 0.15

    # If caller didn't set min_run, use these defaults by mode
    default_min_run_code: int = 2
    default_min_run_text: int = 2

    def decide(
        self,
        *,
        seq_len: int,
        task_mode: str,
        is_sampling: bool,
        requested_quantile: Optional[float],
        requested_min_run: Optional[int],
    ) -> L2Decision:
        if seq_len < self.min_len_enable:
            return L2Decision(
                enabled=False,
                quantile=None,
                min_run=requested_min_run or (self.default_min_run_code if task_mode == "code" else self.default_min_run_text),
                note=f"disabled: seq_len<{self.min_len_enable}",
            )

        # Defaults if not requested
        q = requested_quantile
        if q is None:
            q = self.default_quantile_code if task_mode == "code" else self.default_quantile_text

        mr = requested_min_run
        if mr is None:
            mr = self.default_min_run_code if task_mode == "code" else self.default_min_run_text

        # Clamp aggressiveness during sampling to reduce instability
        if is_sampling and q is not None:
            q = min(float(q), float(self.sampling_quantile_cap))

        return L2Decision(
            enabled=True,
            quantile=float(q) if q is not None else None,
            min_run=int(mr),
            note="enabled",
        )


def l2_policy_from_model_attrs(model) -> L2Policy:
    """
    Convenience: allow runtime tweaks via model attributes without changing configs.
    """
    # If user attaches a custom policy object, use it
    pol = getattr(model, "onn_l2_policy", None)
    if isinstance(pol, L2Policy):
        return pol

    # Otherwise build from optional scalar attributes
    return L2Policy(
        min_len_enable=int(getattr(model, "onn_l2_min_len_enable", 256)),
        sampling_quantile_cap=float(getattr(model, "onn_l2_sampling_quantile_cap", 0.25)),
        default_quantile_code=float(getattr(model, "onn_l2_default_quantile_code", 0.25)),
        default_quantile_text=float(getattr(model, "onn_l2_default_quantile_text", 0.15)),
        default_min_run_code=int(getattr(model, "onn_l2_default_min_run_code", 2)),
        default_min_run_text=int(getattr(model, "onn_l2_default_min_run_text", 2)),
    )