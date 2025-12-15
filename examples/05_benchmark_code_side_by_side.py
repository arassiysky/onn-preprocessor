"""
Side-by-side benchmark (CODE DOMAIN): baseline vs RLE vs ONN-gated (conservative/aggressive)
on long code-oriented prompts.

This is a GPU-correct variant of 04_benchmark_side_by_side.py:
- Uses DEVICE=cuda
- Uses torch.cuda.synchronize() for accurate timing
- Uses code-style prompt expansion
- Sets model.onn_task_mode="code" so compression is disabled by default (safe for code)

Run:
  python examples/05_benchmark_code_side_by_side.py
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
DEVICE = torch.device("cuda")  # RTX 4070
DTYPE = torch.float16          # good default on RTX 4070

WARMUP = 2
REPS = 6

FEATURE_DIM = 8
EPSILON = 0.001
SEED = 42

TARGET_TOKEN_LENGTHS = [128, 256, 512, 1024]

# Code-heavy prompts (Python-centric) designed to be expanded to long contexts.
BASE_PROMPTS = [
    """You are a senior Python engineer.
Write a function `parse_log_lines(lines: list[str]) -> dict[str, int]` that counts HTTP status codes.
Input lines look like: "2025-01-01 GET /api/users 200 123ms".
Return a dict mapping status code strings to counts.
Also write a short docstring and handle malformed lines safely.
Provide a few example inputs and outputs.""",

    """Refactor the following code to be faster and more readable. Keep behavior identical.
Code:
def f(xs):
    out=[]
    for x in xs:
        if x%2==0:
            out.append(x*x)
        else:
            out.append(x+1)
    return out
Now extend it to support numpy arrays efficiently (if numpy is available).
Explain tradeoffs and include tests.""",

    """Debug this Python error. Explain the cause and propose a fix.
Error:
TypeError: 'NoneType' object is not subscriptable
Context:
def get_user(d, k):
    return d.get(k)["name"]
Assume d may contain missing keys or None values.
Provide a corrected implementation and tests.""",

    """Write a clean implementation of an LRU cache in Python.
Requirements:
- O(1) get/put
- Use a doubly linked list + dict
- Include type hints
- Include a minimal usage example
- Include unit tests
Also discuss complexity and common pitfalls.""",
]


@torch.no_grad()
def forward_logits(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**inputs).logits


def _sync_if_cuda(t: torch.Tensor) -> None:
    if t.is_cuda:
        torch.cuda.synchronize()


def time_one(model, inputs: Dict[str, torch.Tensor]) -> float:
    # GPU-correct timing: synchronize around the measured region
    _sync_if_cuda(inputs["input_ids"])
    t0 = time.perf_counter()
    _ = forward_logits(model, inputs)
    _sync_if_cuda(inputs["input_ids"])
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def build_prompt_to_target(tok: AutoTokenizer, base: str, target_len: int) -> Tuple[str, int]:
    """
    Expand a code-oriented prompt until token length >= target_len.
    Uses code-like add-ons to keep the distribution "code assistant" shaped.
    """
    parts: List[str] = []
    cur = ""
    cur_len = 0
    chunk = base.strip()

    addon = (
        "\n\n# Add more edge cases.\n"
        "# Add more tests.\n"
        "# Consider performance and complexity.\n"
        "# Provide one additional alternative implementation.\n"
    )

    while cur_len < target_len:
        parts.append(chunk)
        cur = "\n\n".join(parts)
        cur_len = int(tok(cur, return_tensors="pt")["input_ids"].shape[1])
        chunk = chunk + addon

    return cur, cur_len


def get_comp_len_from_report(model, fallback_len: int) -> int:
    rep = getattr(model, "onn_report", None)
    last = getattr(rep, "last", None) if rep is not None else None
    if not last:
        return fallback_len

    comp = last.get("compression", {})
    # comp can be {"compression": {...}} depending on wrapper nesting
    comp_info = comp.get("compression", comp) if isinstance(comp, dict) else {}
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

    # CODE MODE: safest starting point. Compression is disabled by default for code.
    model.onn_task_mode = "code"
    model.onn_disable_compression_for_code = False  # force compression ON
    model.onn_report_every = 1  # keep at 1 so compressed len is measurable when enabled

    for _ in range(WARMUP):
        _ = forward_logits(model, inputs)

    times = [time_one(model, inputs) for _ in range(REPS)]
    avg_ms = mean(times)

    comp_len = get_comp_len_from_report(model, fallback_len=int(inputs["input_ids"].shape[1]))
    return avg_ms, comp_len


def row(cols, widths):
    print(" | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark is intended for RTX 4070 / GPU.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Loading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    d_model = int(model.config.hidden_size)

    # ---- ONN variants ----
    # NOTE: In code mode, compression is disabled by default by llama_hook.py
    # This first pass isolates embedding augmentation overhead/benefit.
    onn_rle = ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="rle",  # will be disabled in code mode by wrapper unless you override
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
            compression="onn_gated",  # will be disabled in code mode by wrapper unless you override
            onn_gate_quantile=0.25,
            onn_gate_min_run=2,
            onn_gate_threshold=1.4,
        )
    )

    onn_gated_aggr = ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="onn_gated",  # will be disabled in code mode by wrapper unless you override
            onn_gate_quantile=0.35,
            onn_gate_min_run=3,
            onn_gate_threshold=1.4,
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
        base_list, rle_list, q25_list, q35_list = [], [], [], []
        orig_list, comp_rle_list, comp_q25_list, comp_q35_list = [], [], [], []

        for base in BASE_PROMPTS:
            text, _ = build_prompt_to_target(tok, base, target)
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

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
    print("  compRLE  = avg compressed token length using RLE (may be disabled in code mode)")
    print("  compQ25  = avg compressed token length using onn_gated (q=0.25, min_run=2; may be disabled in code mode)")
    print("  compQ35  = avg compressed token length using onn_gated (q=0.35, min_run=3; may be disabled in code mode)")
    print("  *_ms     = avg forward latency in ms (lower is better)")
    print("\nNotes:")
    print("  - This benchmark runs in model.onn_task_mode='code' (safe).")
    print("  - In code mode, llama_hook.py disables compression by default to avoid breaking indentation/syntax.")
    print("  - First goal: measure augmentation overhead on GPU. If acceptable, implement code-safe compression next.")
    print("\nDone.")


if __name__ == "__main__":
    main()