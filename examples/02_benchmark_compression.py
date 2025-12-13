"""
Benchmark Option 1 (compression) + Option 2 (augment) without generation.

Reports:
- seq_in, seq_attn (after compression), ratio
- baseline avg latency
- ONN avg latency
- speedup = base_avg / onn_avg
- determinism (max |A-B|)

Runs two input patterns:
1) random: low compression
2) runs: repeat-heavy, high compression
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, median
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM

from onnpp import ONNConfig, ONNPreprocessor
from onnpp.hf import wrap_llama_for_onn, unwrap_llama_from_onn


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = torch.device("cpu")

SEQ_LENS = [32, 128, 512, 1024]
BATCH_SIZE = 1
WARMUP = 2
REPS = 8

FEATURE_DIM = 8
EPSILON = 0.001
SEED = 42

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
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def time_one(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    t0 = time.perf_counter()
    _ = forward_logits(model, input_ids, attention_mask)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def make_random_ids(seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(1000 + seq_len)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), dtype=torch.int64, generator=g)
    mask = torch.ones((BATCH_SIZE, seq_len), dtype=torch.int64)
    return ids.to(device), mask.to(device)


def make_run_ids(seq_len: int, device: torch.device, run: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Repeat-heavy stream: long runs of the same token id.
    Example run=8: AAAA AAAA BBBB BBBB ...
    """
    g = torch.Generator(device="cpu").manual_seed(2000 + seq_len + run)
    n_blocks = (seq_len + run - 1) // run
    block_ids = torch.randint(0, VOCAB_SIZE, (n_blocks,), dtype=torch.int64, generator=g)
    ids = block_ids.repeat_interleave(run)[:seq_len].view(1, -1)
    mask = torch.ones((1, seq_len), dtype=torch.int64)
    return ids.to(device), mask.to(device)


def row(cols, widths):
    print(" | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))


def benchmark_case(model, onn, input_ids, attention_mask):
    # baseline
    unwrap_llama_from_onn(model)
    for _ in range(WARMUP):
        _ = forward_logits(model, input_ids, attention_mask)
    base_times = [time_one(model, input_ids, attention_mask) for _ in range(REPS)]
    base = Stats(base_times)

    # ONN (compression+augment)
    unwrap_llama_from_onn(model)
    wrap_llama_for_onn(model, onn, store_last_report=True)
    for _ in range(WARMUP):
        _ = forward_logits(model, input_ids, attention_mask)
    onn_times = [time_one(model, input_ids, attention_mask) for _ in range(REPS)]
    onn_stat = Stats(onn_times)

    rep = model.onn_report.last or {}
    shapes = rep.get("shapes", {})
    comp = rep.get("compression", {}).get("compression", rep.get("compression", {}))

    seq_in = input_ids.shape[1]
    seq_attn = shapes.get("input_ids", (None, None))[1] if isinstance(shapes.get("input_ids"), tuple) else None
    if seq_attn is None:
        # fallback to compression report
        seq_attn = comp.get("compressed_len", None)

    ratio = (seq_attn / seq_in) if (seq_attn is not None and seq_in > 0) else None
    speedup = base.avg / onn_stat.avg if onn_stat.avg > 0 else None

    # determinism check
    unwrap_llama_from_onn(model)
    wrap_llama_for_onn(model, onn, store_last_report=False)
    logits_a = forward_logits(model, input_ids, attention_mask)
    unwrap_llama_from_onn(model)
    wrap_llama_for_onn(model, onn, store_last_report=False)
    logits_b = forward_logits(model, input_ids, attention_mask)
    determin = (logits_a - logits_b).abs().max().item()

    return base.avg, onn_stat.avg, speedup, seq_in, seq_attn, ratio, determin


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    d_model = int(model.config.hidden_size)
    cfg = ONNConfig(
        d_model=d_model,
        feature_dim=FEATURE_DIM,
        projector_kind="residual_det",
        epsilon=EPSILON,
        seed=SEED,
        compression="rle",
        rle_min_run=2,
        rle_keep="first",
    )
    onn = ONNPreprocessor(cfg)

    widths = [8, 7, 9, 9, 9, 10, 10]
    print()
    row(["case", "seq_in", "seq_attn", "ratio", "base_avg", "onn_avg", "speedup"], widths)
    row(["-" * w for w in widths], widths)

    for L in SEQ_LENS:
        for case_name, maker in [
            ("random", make_random_ids),
            ("runs8", lambda n, d: make_run_ids(n, d, run=8)),
        ]:
            input_ids, attention_mask = maker(L, DEVICE)
            base_avg, onn_avg, speedup, seq_in, seq_attn, ratio, determin = benchmark_case(
                model, onn, input_ids, attention_mask
            )
            row(
                [
                    case_name,
                    seq_in,
                    seq_attn,
                    f"{ratio:.2f}" if ratio is not None else "n/a",
                    f"{base_avg:.2f}ms",
                    f"{onn_avg:.2f}ms",
                    f"{speedup:.2f}x" if speedup is not None else "n/a",
                ],
                widths,
            )
            if determin != 0:
                print(f"  WARNING: non-determinism detected: max diff {determin}")

    unwrap_llama_from_onn(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
