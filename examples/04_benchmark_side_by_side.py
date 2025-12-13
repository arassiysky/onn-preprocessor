"""
Side-by-side benchmark: baseline vs RLE vs ONN-gated (conservative/aggressive)
on long natural-language prompts.

Run:
  python examples/04_benchmark_side_by_side.py
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp import ONNConfig, ONNPreprocessor
from onnpp.hf import wrap_llama_for_onn, unwrap_llama_from_onn


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = torch.device("cpu")

WARMUP = 2
REPS = 6

FEATURE_DIM = 8
EPSILON = 0.001
SEED = 42

TARGET_TOKEN_LENGTHS = [128, 256, 512, 1024]

BASE_PROMPTS = [
    "Explain in simple terms why attention is O(S^2) and what happens when S grows.",
    "Summarize the idea of token compression and why it can speed up transformers.",
    "Describe a simple plan to learn linear algebra for machine learning in 4 weeks.",
    "List the main components of a transformer and what each does, briefly.",
]


@torch.no_grad()
def forward_logits(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**inputs).logits


def time_one(model, inputs: Dict[str, torch.Tensor]) -> float:
    t0 = time.perf_counter()
    _ = forward_logits(model, inputs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def build_prompt_to_target(tok: AutoTokenizer, base: str, target_len: int) -> Tuple[str, int]:
    """
    Repeat base text until token length >= target_len.
    Adds small variations so the text isn't purely repetitive.
    """
    parts: List[str] = []
    cur = ""
    cur_len = 0
    chunk = base.strip()

    while cur_len < target_len:
        parts.append(chunk)
        cur = " ".join(parts)
        cur_len = int(tok(cur, return_tensors="pt")["input_ids"].shape[1])
        chunk = chunk + " Add one more concrete example and keep it concise."

    return cur, cur_len


def get_comp_len_from_report(model, fallback_len: int) -> int:
    rep = getattr(model, "onn_report", None)
    last = getattr(rep, "last", None) if rep is not None else None
    if not last:
        return fallback_len
    comp = last.get("compression", {})
    # comp can be {"compression": {...}} depending on your wrapper
    comp_info = comp.get("compression", comp)
    return int(comp_info.get("compressed_len", fallback_len))


def bench_baseline(model, inputs) -> float:
    unwrap_llama_from_onn(model)
    for _ in range(WARMUP):
        _ = forward_logits(model, inputs)
    times = [time_one(model, inputs) for _ in range(REPS)]
    return mean(times)


def bench_with_onn(model, onn: ONNPreprocessor, inputs) -> Tuple[float, int]:
    unwrap_llama_from_onn(model)
    wrap_llama_for_onn(model, onn, store_last_report=True)
    for _ in range(WARMUP):
        _ = forward_logits(model, inputs)
    times = [time_one(model, inputs) for _ in range(REPS)]
    avg_ms = mean(times)
    comp_len = get_comp_len_from_report(model, fallback_len=int(inputs["input_ids"].shape[1]))
    return avg_ms, comp_len


def row(cols, widths):
    print(" | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    d_model = int(model.config.hidden_size)

    # ---- ONN variants ----
    onn_rle = ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="rle",
            rle_min_run=2,
            rle_keep="first",
        )
    )

    onn_gated_cons = ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="onn_gated",
            onn_gate_quantile=0.25,
            onn_gate_min_run=2,
            onn_gate_threshold=1.4,  # fallback; unused when quantile != None
        )
    )

    onn_gated_aggr = ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="onn_gated",
            onn_gate_quantile=0.35,
            onn_gate_min_run=3,
            onn_gate_threshold=1.4,  # fallback; unused when quantile != None
        )
    )

    widths = [6, 6, 7, 8, 8, 9, 9, 9, 8]
    print()
    row(
        [
            "tgt",
            "orig",
            "compRLE",
            "compQ25",
            "compQ35",
            "base_ms",
            "rle_ms",
            "q25_ms",
            "q35_ms",
        ],
        widths,
    )
    row(["-" * w for w in widths], widths)

    for target in TARGET_TOKEN_LENGTHS:
        # aggregate over prompts
        base_list, rle_list, q25_list, q35_list = [], [], [], []
        orig_list, comp_rle_list, comp_q25_list, comp_q35_list = [], [], [], []

        for base in BASE_PROMPTS:
            text, orig_len = build_prompt_to_target(tok, base, target)
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            orig_len = int(inputs["input_ids"].shape[1])
            orig_list.append(orig_len)

            base_ms = bench_baseline(model, inputs)
            base_list.append(base_ms)

            rle_ms, comp_rle = bench_with_onn(model, onn_rle, inputs)
            rle_list.append(rle_ms)
            comp_rle_list.append(comp_rle)

            q25_ms, comp_q25 = bench_with_onn(model, onn_gated_cons, inputs)
            q25_list.append(q25_ms)
            comp_q25_list.append(comp_q25)

            q35_ms, comp_q35 = bench_with_onn(model, onn_gated_aggr, inputs)
            q35_list.append(q35_ms)
            comp_q35_list.append(comp_q35)

        # average row
        orig = int(round(mean(orig_list)))
        comp_rle = int(round(mean(comp_rle_list)))
        comp_q25 = int(round(mean(comp_q25_list)))
        comp_q35 = int(round(mean(comp_q35_list)))

        base_ms = mean(base_list)
        rle_ms = mean(rle_list)
        q25_ms = mean(q25_list)
        q35_ms = mean(q35_list)

        row(
            [
                target,
                orig,
                comp_rle,
                comp_q25,
                comp_q35,
                f"{base_ms:.0f}",
                f"{rle_ms:.0f}",
                f"{q25_ms:.0f}",
                f"{q35_ms:.0f}",
            ],
            widths,
        )

    unwrap_llama_from_onn(model)
    print("\nLegend:")
    print("  compRLE  = avg compressed token length using RLE")
    print("  compQ25  = avg compressed token length using onn_gated (quantile=0.25, min_run=2)")
    print("  compQ35  = avg compressed token length using onn_gated (quantile=0.35, min_run=3)")
    print("  *_ms     = avg forward latency in ms (lower is better)")
    print("\nDone.")


if __name__ == "__main__":
    main()