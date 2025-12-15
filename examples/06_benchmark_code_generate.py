"""
Generation benchmark (CODE DOMAIN): baseline vs ONN (adaptive L2) + optional fixed-q sweep.

FINAL ADAPTIVE-ONLY MODE
------------------------
By default this script runs only:
  - base
  - adaptive (compression="onn_gated" but L2 chooses quantile/min_run per prompt)

You can optionally enable a fixed-q sweep for diagnostics by setting:
  ENABLE_Q_SWEEP = True

Output style matches your previous tables:
  columns separated by " | " and a dashed separator row

Run:
  python examples/06_benchmark_code_generate.py
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp import ONNConfig, ONNPreprocessor
from onnpp.hf import wrap_llama_for_onn, unwrap_llama_from_onn


# -----------------------------
# Global config
# -----------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = torch.device("cuda")
DTYPE = torch.float16

SEED = 42
MAX_NEW_TOKENS = 128

WARMUP_PREFILL = 1
WARMUP_GEN = 1         # small generate warmup to avoid first-run kernel/cuda noise
REPS = 4

FEATURE_DIM = 8
EPSILON = 0.001

TARGET_TOKEN_LENGTHS = [128, 256, 512, 1024]

# Final mode: adaptive only (recommended)
ENABLE_Q_SWEEP = False
QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# L2 runtime knobs (adaptive policy)
# You can tune these without changing ONNConfig dataclass (preprocessor uses getattr)
L2_MIN_LEN_ENABLE = 256
L2_MIN_SAVED_FRAC = 0.06
L2_MAX_DROP_FRAC = 0.30
L2_SAMPLING_Q_CAP = 0.25
L2_MIN_RUN_CODE = 2


# -----------------------------
# Prompts (code domain)
# -----------------------------

BASE_PROMPTS = [
    """You are a senior Python engineer.
Write a function `parse_log_lines(lines: list[str]) -> dict[str, int]` that counts HTTP status codes.
Handle malformed lines safely. Include examples and tests.""",
    """Refactor the following code to be faster and more readable.
Code:
def f(xs):
    out=[]
    for x in xs:
        if x%2==0:
            out.append(x*x)
        else:
            out.append(x+1)
    return out
Extend it to support numpy arrays if available.""",
    """Debug this Python error and propose a fix.
Error: TypeError: 'NoneType' object is not subscriptable
Context:
def get_user(d, k):
    return d.get(k)["name"]""",
]


# -----------------------------
# Helpers
# -----------------------------

def _sync() -> None:
    torch.cuda.synchronize()


def build_prompt_to_target(tok: AutoTokenizer, base: str, target_len: int) -> str:
    cur = base.strip()
    addon = (
        "\n\n# Add edge cases.\n"
        "# Add tests.\n"
        "# Discuss complexity.\n"
    )
    while True:
        toks = tok(cur, return_tensors="pt")["input_ids"].shape[1]
        if toks >= target_len:
            return cur
        cur = cur + addon


def get_comp_len(model, fallback: int) -> int:
    rep = getattr(model, "onn_report", None)
    last = getattr(rep, "last", None) if rep else None
    if not last:
        return fallback
    comp = last.get("compression", {})
    comp_info = comp.get("compression", comp)
    return int(comp_info.get("compressed_len", fallback))


def get_l2_note(model) -> str:
    rep = getattr(model, "onn_report", None)
    last = getattr(rep, "last", None) if rep else None
    if not last:
        return ""
    comp = last.get("compression", {})
    comp_info = comp.get("compression", comp)
    note = comp_info.get("l2_note", "")
    return str(note) if note is not None else ""


@torch.no_grad()
def time_prefill(model, inputs) -> float:
    _sync()
    t0 = time.perf_counter()
    _ = model(**inputs).logits
    _sync()
    return (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def time_generate(model, inputs, *, do_sample: bool) -> Tuple[float, float]:
    # Hint to L2 whether this is sampling (used by adaptive policy clamp)
    setattr(model, "onn_is_sampling", bool(do_sample))

    _sync()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        temperature=0.2 if do_sample else None,
        top_p=0.95 if do_sample else None,
        use_cache=True,
    )
    _sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    gen_new = int(out.shape[1] - inputs["input_ids"].shape[1])
    tok_s = (gen_new / (dt_ms / 1000.0)) if dt_ms > 0 else 0.0
    return dt_ms, tok_s


@torch.no_grad()
def warmup_generate(model, inputs) -> None:
    # quick warmup to reduce kernel/cuda first-run outliers
    _sync()
    _ = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
    _sync()


def _attach_runtime_policy_knobs(model) -> None:
    # adaptive L2 knobs
    model.onn_l2_enabled = True
    model.onn_l2_min_len_enable = int(L2_MIN_LEN_ENABLE)
    model.onn_l2_min_saved_frac = float(L2_MIN_SAVED_FRAC)
    model.onn_l2_max_drop_frac = float(L2_MAX_DROP_FRAC)
    model.onn_l2_sampling_quantile_cap = float(L2_SAMPLING_Q_CAP)
    model.onn_l2_min_run_code = int(L2_MIN_RUN_CODE)

    # L4 toggle (keep enabled for final demo; switch off if you want pure L2)
    model.onn_l4_enabled = True


# -----------------------------
# Benchmark core
# -----------------------------

def bench_variant(
    model,
    inputs,
    *,
    mode: str,
    q: float | None = None,
) -> Dict[str, float | int | str]:
    """
    mode:
      - "base": no ONN
      - "adaptive": ONN with compression="onn_gated"; L2 decides quantile/min_run per prompt
      - "qXX": ONN with fixed ONNConfig.onn_gate_quantile set, but L2 may still override (depends on your policy)
    """
    unwrap_llama_from_onn(model)

    if mode != "base":
        # Build ONN config
        cfg = ONNConfig(
            d_model=int(model.config.hidden_size),
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="onn_gated",
            onn_gate_min_run=2,
        )
        if q is not None:
            cfg = ONNConfig(
                d_model=int(model.config.hidden_size),
                feature_dim=FEATURE_DIM,
                projector_kind="residual_det",
                epsilon=EPSILON,
                seed=SEED,
                compression="onn_gated",
                onn_gate_quantile=float(q),
                onn_gate_min_run=2,
            )

        onn = ONNPreprocessor(cfg)
        wrap_llama_for_onn(model, onn, store_last_report=True)

        # Ensure compression is allowed in code mode
        model.onn_task_mode = "code"
        model.onn_disable_compression_for_code = False
        model.onn_report_every = 1

        _attach_runtime_policy_knobs(model)

    # Warmup (prefill)
    for _ in range(WARMUP_PREFILL):
        _ = model(**inputs).logits

    # Optional warmup generate to reduce outliers
    for _ in range(WARMUP_GEN):
        warmup_generate(model, inputs)

    prefill_ms = float(mean(time_prefill(model, inputs) for _ in range(REPS)))
    orig_len = int(inputs["input_ids"].shape[1])
    comp_len = int(get_comp_len(model, orig_len))
    l2_note = get_l2_note(model)

    greedy_pairs = [time_generate(model, inputs, do_sample=False) for _ in range(REPS)]
    greedy_ms = float(mean(p[0] for p in greedy_pairs))
    greedy_tok_s = float(mean(p[1] for p in greedy_pairs))

    samp_pairs = [time_generate(model, inputs, do_sample=True) for _ in range(REPS)]
    samp_ms = float(mean(p[0] for p in samp_pairs))
    samp_tok_s = float(mean(p[1] for p in samp_pairs))

    unwrap_llama_from_onn(model)

    return {
        "orig": orig_len,
        "comp": comp_len,
        "prefill_ms": prefill_ms,
        "greedy_ms": greedy_ms,
        "g_tok_s": greedy_tok_s,
        "samp_ms": samp_ms,
        "s_tok_s": samp_tok_s,
        "l2_note": l2_note,
    }


# -----------------------------
# Printing (aligned)
# -----------------------------

COLS = [
    ("tgt", 6),
    ("orig", 6),
    ("comp", 6),
    ("prefill_ms", 10),
    ("greedy_ms", 12),
    ("g_tok/s", 10),
    ("samp_ms", 12),
    ("s_tok/s", 10),
    ("variant", 10),
]

# Print L2 note optionally (kept off by default to preserve classic table style)
PRINT_L2_NOTE = False
L2_NOTE_COL = ("l2_note", 28)


def row(values: List[str]) -> None:
    widths = [w for _, w in COLS] + ([L2_NOTE_COL[1]] if PRINT_L2_NOTE else [])
    out = []
    for v, w in zip(values, widths):
        out.append(v.ljust(w))
    print(" | ".join(out))


def dash_row() -> None:
    widths = [w for _, w in COLS] + ([L2_NOTE_COL[1]] if PRINT_L2_NOTE else [])
    print(" | ".join("-" * w for w in widths))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    # Header
    header = [name for name, _ in COLS]
    if PRINT_L2_NOTE:
        header.append(L2_NOTE_COL[0])
    row(header)
    dash_row()

    variants: List[Tuple[str, float | None]] = [("base", None), ("adaptive", None)]
    if ENABLE_Q_SWEEP:
        variants += [(f"q{int(q*100)}", q) for q in QUANTILES]

    for tgt in TARGET_TOKEN_LENGTHS:
        for vname, q in variants:
            rows: List[Dict[str, float | int | str]] = []

            for base in BASE_PROMPTS:
                text = build_prompt_to_target(tok, base, tgt)
                inputs = tok(text, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                if vname == "base":
                    rows.append(bench_variant(model, inputs, mode="base", q=None))
                elif vname == "adaptive":
                    rows.append(bench_variant(model, inputs, mode="adaptive", q=None))
                else:
                    rows.append(bench_variant(model, inputs, mode="q", q=q))

            # Aggregate across prompts
            # (keep l2_note blank in aggregate unless PRINT_L2_NOTE True; if enabled, show most common note)
            agg_num = {}
            for k in ("orig", "comp", "prefill_ms", "greedy_ms", "g_tok_s", "samp_ms", "s_tok_s"):
                agg_num[k] = mean(float(r[k]) for r in rows)

            l2_note = ""
            if PRINT_L2_NOTE:
                notes = [str(r.get("l2_note", "")) for r in rows]
                # choose the longest non-empty note (simple, stable)
                notes = [n for n in notes if n]
                l2_note = max(notes, key=len) if notes else ""

            values = [
                f"{tgt}",
                f"{int(agg_num['orig'])}",
                f"{int(agg_num['comp'])}",
                f"{agg_num['prefill_ms']:.0f}",
                f"{agg_num['greedy_ms']:.0f}",
                f"{agg_num['g_tok_s']:.1f}",
                f"{agg_num['samp_ms']:.0f}",
                f"{agg_num['s_tok_s']:.1f}",
                vname,
            ]
            if PRINT_L2_NOTE:
                values.append(l2_note[: L2_NOTE_COL[1] - 1])

            row(values)


if __name__ == "__main__":
    main()