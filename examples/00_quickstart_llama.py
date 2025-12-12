"""
Quickstart: ONN Pre-Processor (v0.L) embedding augmentation with a Llama-family model.

This example:
  1) Loads a small Llama-family model (TinyLlama)
  2) Runs a baseline forward pass
  3) Wraps the model with ONN (augment embeddings before attention)
  4) Runs a second forward pass
  5) Prints shapes + ONN report summary + timing

Run:
  python examples/00_quickstart_llama.py
"""

from __future__ import annotations

import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp import ONNConfig, ONNPreprocessor
from onnpp.hf import wrap_llama_for_onn

# ONN mode:
#   "safe"   -> projector_init="identity_like" (E_aug ≈ E, logits nearly identical)
#   "active" -> projector_init="xavier"        (ONN features influence embeddings)
ONN_MODE = "safe"   # change to "active" to see effect

def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


@torch.no_grad()
def run_forward(model, inputs) -> torch.Tensor:
    out = model(**inputs)
    # logits: (B, S, vocab)
    return out.logits


def summarize_report(report: Dict[str, Any]) -> str:
    shapes = report.get("shapes", {})
    cfg = report.get("config", {})
    f = report.get("featurizer", {})
    p = report.get("projector", {})
    return (
        f"ONN report:\n"
        f"  version: {report.get('onnpp_version')}\n"
        f"  d_model: {cfg.get('d_model')} | feature_dim: {cfg.get('feature_dim')}\n"
        f"  shapes: {shapes}\n"
        f"  featurizer: {f.get('featurizer')}\n"
        f"  projector: {p.get('projector')} (in={p.get('in_dim')} -> out={p.get('out_dim')})\n"
    )


def main() -> None:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "Hello world."

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Keep CPU by default (simple + reproducible). If you want GPU:
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up (optional but helps timing stability)
    _ = run_forward(model, inputs)

    # Baseline timing
    t0 = time.perf_counter()
    logits_base = run_forward(model, inputs)
    t1 = time.perf_counter()

    print("\nBaseline:")
    print("  logits shape:", tuple(logits_base.shape))
    print(f"  forward time: {_ms(t0, t1):.2f} ms")

    # Build ONN v0.L config from model hidden size
    d_model = int(model.config.hidden_size)
    if ONN_MODE not in ("safe", "active"):
        raise ValueError("ONN_MODE must be 'safe' or 'active'")

    projector_init = "identity_like" if ONN_MODE == "safe" else "xavier"

    cfg = ONNConfig(
        d_model=d_model,
        feature_dim=8,
        projector_init=projector_init,
    )

    onn = ONNPreprocessor(cfg)
    print(f"\nONN_MODE={ONN_MODE} (projector_init={projector_init})")


    # Wrap model with ONN
    wrap_llama_for_onn(model, onn, store_last_report=True)

    # Warm-up ONN path
    _ = run_forward(model, inputs)

    # ONN timing
    t2 = time.perf_counter()
    logits_onn = run_forward(model, inputs)
    t3 = time.perf_counter()

    print("\nWith ONN (v0.L augment):")
    print("  logits shape:", tuple(logits_onn.shape))
    print(f"  forward time: {_ms(t2, t3):.2f} ms")

    # Compare outputs lightly (not expecting equality because embeddings changed)
    delta = (logits_onn - logits_base).abs().mean().item()
    print(f"\nMean |logits_onn - logits_base|: {delta:.6f}")

    # Print ONN report summary
    if hasattr(model, "onn_report") and model.onn_report and model.onn_report.last:
        print("\n" + summarize_report(model.onn_report.last))
    else:
        print("\nNo ONN report found (store_last_report=False?).")

    print("Done.")


if __name__ == "__main__":
    main()
