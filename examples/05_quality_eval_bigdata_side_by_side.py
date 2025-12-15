from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp.hf import wrap_llama_for_onn, unwrap_llama_from_onn
from onnpp.api.preprocessor import ONNPreprocessor, ONNConfig

# ---- match 04 defaults ----
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = torch.device("cpu")

# These should match your 04 file constants (copy them if they differ)
FEATURE_DIM = 256
EPSILON = 1e-6
SEED = 123
DTYPE = torch.float32

MAX_NEW_TOKENS = 256
DO_SAMPLE = False  # deterministic
# ---------------------------

BIGDATA_TEXT = """\
Welcome to Big Data For Dummies. Big data is becoming one of the
most important technology trends that has the potential for dramatically
changing the way organizations use information to enhance the customer
experience and transform their business models. How does a company
go about using data to the best advantage? What does it mean to transform
massive amounts of data into knowledge? In this book, we provide you with
insights into how technology transitions in software, hardware, and delivery
models are changing the way that data can be used in new ways.
Big data is not a single market. Rather, it is a combination of data-management
technologies that have evolved over time. Big data enables organizations
to store, manage, and manipulate vast amounts of data at the right
speed and at the right time to gain the right insights. The key to understanding
big data is that data has to be managed so that it can meet the business
requirement a given solution is designed to support. Most companies are at
an early stage with their big data journey. Many companies are experimenting
with techniques that allow them to collect massive amounts of data to
determine whether hidden patterns exist within that data that might be an
early indication of an important change. Some data may indicate that customer
buying patterns are changing or that new elements are in the business
that need to be addressed before it is too late.
As companies begin to evaluate new types of big data solutions, many new
opportunities will unfold. For example, manufacturing companies may be
able to monitor data coming from machine sensors to determine how processes
need to be modified before a catastrophic event happens. It will be
possible for retailers to monitor data in real time to upsell customers related
products as they are executing a transaction. Big data solutions can be used
in healthcare to determine the cause of an illness and provide a physician
with guidance on treatment options.
Big data is not an isolated solution, however. Implementing a big data solution
requires that the infrastructure be in place to support the scalability,
distribution, and management of that data. Therefore, it is important to put
both a business and technical strategy in place to make use of this important
technology trend.
For many important reasons, we think that it is important for you to understand
big data technologies and know the ways that companies are using
emerging technologies such as Hadoop, MapReduce, and new database engines to
transform the value of their data. We wrote this book to provide a perspective
on what big data is and how it’s changing the way that organizations can leverage
more data than was possible in the past. We think that this book will give you the
context to make informed decisions.
"""

PROBES: List[Tuple[str, str]] = [
    ("P1_definition",
     "According to the text, what is big data? Answer using only information explicitly stated in the text."),
    ("P2_industries",
     "List all industries explicitly mentioned in the text as examples of where big data can be applied. Return only a comma-separated list."),
    ("P3_why_experiment",
     "Why do companies experiment with collecting massive amounts of data, according to the text?"),
    ("P4_unanswerable_security",
     "According to the text, what are the security risks of big data systems? If the text does not mention this, say: NOT IN TEXT."),
    ("P5_quote_infra",
     "Quote one sentence from the text that explains why infrastructure is important for big data solutions. Return only the quote."),
]


def build_prompt(doc: str, question: str) -> str:
    return (
        "You are given a source text. Answer the question using ONLY the source text.\n"
        "If the answer is not in the text, say: NOT IN TEXT.\n\n"
        f"SOURCE TEXT:\n{doc}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )


def normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def score_industries(output: str) -> float:
    gt = {"manufacturing", "retail", "healthcare"}
    s = output.lower()
    items = re.split(r"[,\n;]+", s)
    items = {re.sub(r"[^a-z]+", "", x).strip() for x in items}
    items = {x for x in items if x}
    return 1.0 if items == gt else 0.0


def score_unanswerable(output: str) -> float:
    s = output.strip().lower()
    if "not in text" in s:
        return 1.0
    if "does not" in s and ("mention" in s or "discuss" in s) and "security" in s:
        return 1.0
    return 0.0


def score_quote_in_text(output: str, doc: str) -> float:
    o = normalize_spaces(output)
    d = normalize_spaces(doc)
    if len(o) < 20:
        return 0.0
    return 1.0 if o in d else 0.0


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@dataclass
class ProbeResult:
    pid: str
    output: str
    out_hash: str
    industries_exact: float | None = None
    not_in_text: float | None = None
    quote_in_text: float | None = None


def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)
    model.config.use_cache = False
    model.eval()
    return model, tok


def build_preprocessors(d_model: int) -> Dict[str, ONNPreprocessor]:
    # Copied directly from your snippet (q25/q35 are gate quantiles)
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

    onn_gated_q25 = ONNPreprocessor(
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

    onn_gated_q35 = ONNPreprocessor(
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

    return {
        "baseline": None,        # type: ignore
        "onn_rle": onn_rle,
        "onn_gated_q25": onn_gated_q25,
        "onn_gated_q35": onn_gated_q35,
    }


def run_variant(model, tok, name: str, onn: ONNPreprocessor | None) -> Dict[str, ProbeResult]:
    if onn is None:
        unwrap_llama_from_onn(model)
    else:
        unwrap_llama_from_onn(model)
        wrap_llama_for_onn(model, onn, store_last_report=True)
        model.config.use_cache = False

    results: Dict[str, ProbeResult] = {}
    for pid, question in PROBES:
        prompt = build_prompt(BIGDATA_TEXT, question)
        out = generate_answer(model, tok, prompt)

        pr = ProbeResult(pid=pid, output=out, out_hash=short_hash(out))
        if pid == "P2_industries":
            pr.industries_exact = score_industries(out)
        if pid == "P4_unanswerable_security":
            pr.not_in_text = score_unanswerable(out)
        if pid == "P5_quote_infra":
            pr.quote_in_text = score_quote_in_text(out, BIGDATA_TEXT)

        results[pid] = pr

    return results


def print_summary(variant: str, suite: Dict[str, ProbeResult]) -> None:
    print(f"\n=== {variant} ===")
    print(f"{'probe':24} | {'hash':12} | {'industries':10} | {'not_in_text':10} | {'quote_in_text':12}")
    print("-" * 82)
    for pid, _ in PROBES:
        pr = suite[pid]
        print(
            f"{pid:24} | {pr.out_hash:12} | "
            f"{'' if pr.industries_exact is None else f'{pr.industries_exact:.2f}':10} | "
            f"{'' if pr.not_in_text is None else f'{pr.not_in_text:.2f}':10} | "
            f"{'' if pr.quote_in_text is None else f'{pr.quote_in_text:.2f}':12}"
        )


def print_outputs(variant: str, suite: Dict[str, ProbeResult]) -> None:
    for pid, _ in PROBES:
        pr = suite[pid]
        print(f"\n--- {variant} / {pid} ---")
        print(pr.output)


def main():
    model, tok = load_model_and_tokenizer()

    # Get d_model from the model config (matches your 04 pattern)
    d_model = int(getattr(model.config, "hidden_size"))
    preprocessors = build_preprocessors(d_model)

    suites: Dict[str, Dict[str, ProbeResult]] = {}
    for name, onn in preprocessors.items():
        suites[name] = run_variant(model, tok, name, onn)

    for name, suite in suites.items():
        print_summary(name, suite)

    # Full outputs for manual inspection
    for name, suite in suites.items():
        print_outputs(name, suite)

    unwrap_llama_from_onn(model)


if __name__ == "__main__":
    main()