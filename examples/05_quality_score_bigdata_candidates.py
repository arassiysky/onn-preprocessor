from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp.api.preprocessor import ONNPreprocessor, ONNConfig


# ----------------- Match your 04 defaults -----------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = torch.device("cpu")

FEATURE_DIM = 256
EPSILON = 1e-6
SEED = 123
DTYPE = torch.float32
# ----------------------------------------------------------


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


def build_prompt(doc: str, question: str) -> str:
    return (
        "You are given a source text. Answer the question using ONLY the source text.\n"
        "If the answer is not in the text, say: NOT IN TEXT.\n\n"
        f"SOURCE TEXT:\n{doc}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )


# ----------------- Candidate probes -----------------
PROBES: List[Tuple[str, str, str, List[str]]] = [
    (
        "P2_industries",
        "List all industries explicitly mentioned in the text as examples of where big data can be applied. Return only a comma-separated list.",
        "manufacturing, retail, healthcare",
        [
            "finance, retail, healthcare",
            "manufacturing, telecom, healthcare",
            "manufacturing, retail, education",
            "retail, healthcare",
        ],
    ),
    (
        "P4_unanswerable_security",
        "According to the text, what are the security risks of big data systems? If the text does not mention this, say: NOT IN TEXT.",
        "NOT IN TEXT",
        [
            # Hallucination variants with different styles
            "Security risks include data breaches, unauthorized access, and compliance violations.",
            "Big data systems face privacy risks due to large-scale data aggregation.",
            "Security concerns arise from storing massive datasets across distributed systems.",
            "Big data introduces cybersecurity risks such as insider threats and external attacks.",
            "Risks include exposure of sensitive information and lack of proper access controls.",
        ],
    ),
    (
        "P5_quote_infra",
        "Quote one sentence from the text that explains why infrastructure is important for big data solutions. Return only the quote.",
        "Implementing a big data solution requires that the infrastructure be in place to support the scalability, distribution, and management of that data.",
        [
            "Implementing big data requires infrastructure that supports scalability and distribution.",
            "Infrastructure is important because big data must be stored and managed at scale.",
            "Big data solutions depend on infrastructure for scalability and management.",
        ],
    ),
]
# ----------------------------------------------------


def _to_int_list(x: Any) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if torch.is_tensor(x):
        return [int(v) for v in x.flatten().tolist()]
    raise TypeError(f"Unsupported token container type: {type(x)}")


def compress_ids(onn: ONNPreprocessor, ids: List[int]) -> List[int]:
    """
    compress_tokens expects a torch tensor (batch, seq).
    Returns list[int] of compressed token ids for batch item 0.
    Supports returns:
      - tensor (B, S')
      - tuple(tensor, report)
      - dict containing a tensor/list under common keys
    """
    ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # [1, S]
    out = onn.compress_tokens(ids_t)

    # Helper to normalize to list[int] for batch item 0
    def tensor_row_to_list(x: Any) -> List[int]:
        if torch.is_tensor(x):
            if x.dim() == 2:
                x = x[0]
            return [int(v) for v in x.flatten().tolist()]
        if isinstance(x, list):
            # could be [ids...] or [[ids...]]
            if len(x) > 0 and isinstance(x[0], list):
                x = x[0]
            return [int(v) for v in x]
        raise TypeError(type(x))

    # tensor
    if torch.is_tensor(out) or isinstance(out, list):
        return tensor_row_to_list(out)

    # tuple(tensor, ...)
    if isinstance(out, tuple) and len(out) >= 1:
        return tensor_row_to_list(out[0])

    # dict
    if isinstance(out, dict):
        for k in ("tokens", "input_ids", "ids", "compressed_tokens", "compressed_ids"):
            if k in out:
                return tensor_row_to_list(out[k])

    raise TypeError(f"Unexpected compress_tokens return type: {type(out)}")


def longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    L = min(len(a), len(b))
    i = 0
    while i < L and a[i] == b[i]:
        i += 1
    return i


@torch.no_grad()
def mean_logprob_of_suffix(model, full_ids: List[int], prefix_len: int) -> float:
    """
    full_ids: token ids that the model will actually see
    prefix_len: how many of those tokens are "prompt" (the rest is continuation)
    Returns mean logprob per continuation token.
    """
    if prefix_len <= 0:
        return -math.inf
    if prefix_len >= len(full_ids):
        return -math.inf

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
    outputs = model(input_ids)
    logits = outputs.logits  # [1, seq, vocab]

    # continuation labels are from prefix_len .. end-1
    labels = input_ids[:, prefix_len:]  # [1, cont_len]
    cont_len = labels.shape[1]
    if cont_len <= 0:
        return -math.inf

    # logits predicting those labels are at positions prefix_len-1 .. end-2
    logits_slice = logits[:, prefix_len - 1 : -1, :]  # [1, cont_len, vocab]
    if logits_slice.shape[1] != cont_len:
        return -math.inf

    logprobs = torch.log_softmax(logits_slice, dim=-1)
    token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, cont_len]
    return float(token_logprobs.mean().item())


@torch.no_grad()
def score_candidate(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    onn: Optional[ONNPreprocessor],
) -> float:
    """
    Scores candidate under either baseline (onn=None) or ONN preprocessing (onn!=None).
    Uses mean logprob per continuation token in the *actual* token space seen by model.

    Key trick: when ONN is enabled, we compress:
      - prompt_ids
      - full_ids = prompt_ids + candidate_ids
    Then define prefix_len by longest-common-prefix in compressed space.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cand_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + cand_ids

    if onn is None:
        prefix_len = len(prompt_ids)
        return mean_logprob_of_suffix(model, full_ids, prefix_len)

    comp_prompt = compress_ids(onn, prompt_ids)
    comp_full = compress_ids(onn, full_ids)

    prefix_len = longest_common_prefix_len(comp_prompt, comp_full)
    return mean_logprob_of_suffix(model, comp_full, prefix_len)


def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)
    model.eval()
    return model, tok


def build_preprocessors(d_model: int) -> Dict[str, Optional[ONNPreprocessor]]:
    # Copied from your snippet
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
            onn_gate_threshold=1.4,
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
            onn_gate_threshold=1.4,
        )
    )

    return {
        "baseline": None,
        "onn_rle": onn_rle,
        "onn_gated_q25": onn_gated_q25,
        "onn_gated_q35": onn_gated_q35,
    }


@dataclass
class ProbeScore:
    probe_id: str
    is_correct: bool
    chosen: str
    correct: str
    chosen_score: float
    correct_score: float


def run_variant(model, tok, vname: str, onn: Optional[ONNPreprocessor]) -> List[ProbeScore]:
    results: List[ProbeScore] = []

    for pid, question, correct, distractors in PROBES:
        prompt = build_prompt(BIGDATA_TEXT, question)
        candidates = [correct] + distractors

        scores = [score_candidate(model, tok, prompt, c, onn) for c in candidates]
        best_i = max(range(len(candidates)), key=lambda i: scores[i])

        chosen = candidates[best_i]
        results.append(
            ProbeScore(
                probe_id=pid,
                is_correct=(best_i == 0),
                chosen=chosen,
                correct=correct,
                chosen_score=scores[best_i],
                correct_score=scores[0],
            )
        )

    return results


def main():
    model, tok = load_model_and_tokenizer()
    d_model = int(getattr(model.config, "hidden_size"))

    variants = build_preprocessors(d_model)

    print("\nCandidate-scoring quality eval (fast, no generation)")
    print("Higher mean logprob = model prefers that answer.\n")

    for vname, onn in variants.items():
        scores = run_variant(model, tok, vname, onn)
        ok = sum(1 for s in scores if s.is_correct)
        acc = ok / len(scores)

        print(f"=== {vname} ===  accuracy={acc:.2f} ({ok}/{len(scores)})")
        for s in scores:
            print(f"- {s.probe_id}: {'OK' if s.is_correct else 'FAIL'}")
            if not s.is_correct:
                print(f"  chosen : {s.chosen}")
                print(f"  correct: {s.correct}")
            print(f"  scores: chosen={s.chosen_score:.3f}  correct={s.correct_score:.3f}")
        print("")


if __name__ == "__main__":
    main()
