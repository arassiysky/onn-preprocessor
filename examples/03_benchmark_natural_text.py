"""
Benchmark Option 1.5 (onn_gated quantile compression) on long natural-language text.

It constructs long prompts by repeating base prompts until reaching a target token length,
then benchmarks baseline vs ONN, and reports compression ratio + speedup.

Run:
  python examples/03_benchmark_natural_text.py
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

# Option 1.5 settings
GATE_QUANTILE = 0.25
MIN_RUN = 2
FALLBACK_THRESHOLD = 1.4  # unused when quantile != None, but kept for transparency

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
    Repeat `base` until token length >= target_len (then stop).
    Returns (text, token_len).
    """
    parts: List[str] = []
    cur = ""
    cur_len = 0

    # grow in chunks to avoid too many tokenizations
    chunk = base.strip()
    while cur_len < target_len:
        parts.append(chunk)
        cur = " ".join(parts)
        cur_len = int(tok(cur, return_tensors="pt")["input_ids"].shape[1])

        # small variation to avoid extreme repetitiveness
        chunk = chunk + " " + "Add one more concrete example and keep it concise."

    return cur, cur_len


def row(cols, widths):
    print(" | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    d_model = int(model.config.hidden_size)
    cfg = ONNConfig(
        d_model=d_model,
        feature_dim=FEATURE_DIM,
        projector_kind="residual_det",
        epsilon=EPSILON,
        seed=SEED,
        compression="onn_gated",
        onn_gate_quantile=GATE_QUANTILE,
        onn_gate_min_run=MIN_RUN,
        onn_gate_threshold=FALLBACK_THRESHOLD,
    )
    onn = ONNPreprocessor(cfg)

    widths = [8, 8, 10, 7, 11, 11, 8]
    print()
    row(["target", "orig", "comp", "ratio", "base_avg", "onn_avg", "speedup"], widths)
    row(["-" * w for w in widths], widths)

    # For each target length, average over multiple base prompts
    for target in TARGET_TOKEN_LENGTHS:
        rows: List[Tuple[int, int, float, float]] = []

        for base in BASE_PROMPTS:
            text, orig_len = build_prompt_to_target(tok, base, target)
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Baseline
            unwrap_llama_from_onn(model)
            for _ in range(WARMUP):
                _ = forward_logits(model, inputs)
            base_times = [time_one(model, inputs) for _ in range(REPS)]
            base_ms = mean(base_times)

            # ONN (stores report)
            unwrap_llama_from_onn(model)
            wrap_llama_for_onn(model, onn, store_last_report=True)
            for _ in range(WARMUP):
                _ = forward_logits(model, inputs)
            onn_times = [time_one(model, inputs) for _ in range(REPS)]
            onn_ms = mean(onn_times)

            rep = model.onn_report.last or {}
            comp = rep.get("compression", {}).get("compression", rep.get("compression", {}))
            comp_len = int(comp.get("compressed_len", orig_len))

            rows.append((orig_len, comp_len, base_ms, onn_ms))

        # Aggregate across prompts
        avg_orig = int(round(mean([r[0] for r in rows])))
        avg_comp = int(round(mean([r[1] for r in rows])))
        avg_ratio = mean([(r[1] / r[0]) for r in rows if r[0] > 0])
        avg_base = mean([r[2] for r in rows])
        avg_onn = mean([r[3] for r in rows])
        avg_speedup = avg_base / avg_onn if avg_onn > 0 else 1.0

        row(
            [
                target,
                avg_orig,
                avg_comp,
                f"{avg_ratio:.2f}",
                f"{avg_base:.2f}ms",
                f"{avg_onn:.2f}ms",
                f"{avg_speedup:.2f}x",
            ],
            widths,
        )

    unwrap_llama_from_onn(model)
    print("\nDone.")


if __name__ == "__main__":
    main()