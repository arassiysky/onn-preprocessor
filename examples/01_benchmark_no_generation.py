"""
Benchmark v0.L (Option 2: augment embeddings) without generation.

Measures:
- baseline forward latency
- ONN forward latency (deterministic residual projector)
- overhead %
- mean absolute logits delta (sanity)
- determinism check (ONN run A vs B)

Run:
  python examples/01_benchmark_no_generation.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM

from onnpp import ONNConfig, ONNPreprocessor
from onnpp.hf import wrap_llama_for_onn, unwrap_llama_from_onn


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Device policy (keep CPU for reproducibility; you can switch to CUDA later)
DEVICE = torch.device("cpu")

# Benchmark settings
SEQ_LENS = [8, 32, 128, 512]     # you can extend to 1024/2048 if RAM allows
BATCH_SIZE = 1
WARMUP = 2
REPS = 8

# ONN deterministic settings
FEATURE_DIM = 8
EPSILON = 0.001
SEED = 42

# Fixed vocab size for synthetic inputs (TinyLlama vocab is 32000)
VOCAB_SIZE = 32000


@dataclass
class Stats:
    times_ms: List[float]

    @property
    def avg(self) -> float:
        return mean(self.times_ms)

    @property
    def med(self) -> float:
        return median(self.times_ms)


@torch.no_grad()
def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


def time_one(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    t0 = time.perf_counter()
    _ = forward_logits(model, input_ids, attention_mask)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def make_synth_batch(seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic token IDs so we can control length precisely.
    """
    g = torch.Generator(device="cpu").manual_seed(1234 + seq_len)
    ids = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, seq_len), dtype=torch.int64, generator=g)
    mask = torch.ones((BATCH_SIZE, seq_len), dtype=torch.int64)
    return ids.to(device), mask.to(device)


def print_row(cols: List[str], widths: List[int]) -> None:
    line = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    d_model = int(model.config.hidden_size)

    # Build ONN (deterministic residual projector)
    cfg = ONNConfig(
        d_model=d_model,
        feature_dim=FEATURE_DIM,
        projector_kind="residual_det",
        epsilon=EPSILON,
        seed=SEED,
    )
    onn = ONNPreprocessor(cfg)

    # Header
    widths = [7, 12, 12, 10, 14, 14]
    print()
    print_row(
        ["seq", "base_avg", "onn_avg", "overhead", "logits_delta", "onn_determin"],
        widths,
    )
    print_row(["-" * w for w in widths], widths)

    for L in SEQ_LENS:
        input_ids, attention_mask = make_synth_batch(L, DEVICE)

        # ---- Baseline ----
        # Ensure model is unwrapped
        unwrap_llama_from_onn(model)

        # Warmup
        for _ in range(WARMUP):
            _ = forward_logits(model, input_ids, attention_mask)

        base_times = [time_one(model, input_ids, attention_mask) for _ in range(REPS)]
        base = Stats(base_times)

        # ---- ONN ----
        unwrap_llama_from_onn(model)
        wrap_llama_for_onn(model, onn, store_last_report=False)

        for _ in range(WARMUP):
            _ = forward_logits(model, input_ids, attention_mask)

        onn_times = [time_one(model, input_ids, attention_mask) for _ in range(REPS)]
        onn_stat = Stats(onn_times)

        # ---- Delta sanity (single pass) ----
        unwrap_llama_from_onn(model)
        logits_base = forward_logits(model, input_ids, attention_mask)

        unwrap_llama_from_onn(model)
        wrap_llama_for_onn(model, onn, store_last_report=False)
        logits_onn = forward_logits(model, input_ids, attention_mask)

        logits_delta = (logits_onn - logits_base).abs().mean().item()

        # ---- Determinism check: ONN run A vs ONN run B ----
        unwrap_llama_from_onn(model)
        wrap_llama_for_onn(model, onn, store_last_report=False)
        logits_a = forward_logits(model, input_ids, attention_mask)

        unwrap_llama_from_onn(model)
        wrap_llama_for_onn(model, onn, store_last_report=False)
        logits_b = forward_logits(model, input_ids, attention_mask)

        determin = (logits_a - logits_b).abs().max().item()  # should be 0 (or extremely tiny)

        # ---- Overhead ----
        overhead_pct = ((onn_stat.avg / base.avg) - 1.0) * 100.0

        print_row(
            [
                str(L),
                f"{base.avg:.2f}ms",
                f"{onn_stat.avg:.2f}ms",
                f"{overhead_pct:+.1f}%",
                f"{logits_delta:.6f}",
                f"{determin:.3g}",
            ],
            widths,
        )

    # Clean up
    unwrap_llama_from_onn(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
