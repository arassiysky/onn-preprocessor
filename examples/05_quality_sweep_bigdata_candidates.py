from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp.api.preprocessor import ONNPreprocessor, ONNConfig


# ----------------- Hardware / dtype -----------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 40-series likes TF32 for matmul; helps speed without affecting our "which candidate wins" much
if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True

DTYPE = torch.float16 if USE_CUDA else torch.float32
# ----------------------------------------------------


# ----------------- Model / ONN constants -----------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

FEATURE_DIM = 256
EPSILON = 1e-6
SEED = 123
# ----------------------------------------------------------


# Sweep values (edit freely)
GATE_QUANTILES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


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


# probe_id, kind, question, correct, distractors
# kind: "A" = answerable (grounding), "U" = unanswerable (refusal)
PROBES: List[Tuple[str, str, str, str, List[str]]] = [
    # ---------- Answerable / grounding ----------
    (
        "A1_industries",
        "A",
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
        "A2_tech_names",
        "A",
        "Name the emerging technologies explicitly listed in the text. Return only a comma-separated list.",
        "Hadoop, MapReduce, new database engines",
        [
            "Hadoop, Spark, NoSQL databases",
            "MapReduce, Kafka, Hadoop",
            "Hadoop, MapReduce, blockchain",
            "Spark, NoSQL, Hadoop",
        ],
    ),
    (
        "A3_not_single_market",
        "A",
        "According to the text, is big data a single market? Answer in one short sentence.",
        "Big data is not a single market; it is a combination of data-management technologies that have evolved over time.",
        [
            "Big data is a single market focused on analytics platforms.",
            "Big data is primarily one market driven by cloud storage providers.",
            "Big data is a single market centered on Hadoop ecosystems.",
        ],
    ),
    (
        "A4_why_experiment",
        "A",
        "Why do companies experiment with collecting massive amounts of data, according to the text?",
        "To determine whether hidden patterns exist within the data that might be an early indication of an important change.",
        [
            "To train artificial intelligence models on proprietary datasets.",
            "To reduce operating costs through automation and process replacement.",
            "To comply with regulatory reporting requirements.",
        ],
    ),
    (
        "A5_infra_reason",
        "A",
        "According to the text, why is infrastructure important for implementing a big data solution?",
        "Because the infrastructure must support the scalability, distribution, and management of the data.",
        [
            "Because infrastructure is needed mainly for encryption and security monitoring.",
            "Because infrastructure is required primarily to reduce latency for gaming workloads.",
            "Because infrastructure is needed to store data only, not to manage or distribute it.",
        ],
    ),

    # ---------- Unanswerable / refusal ----------
    (
        "U1_security_risks",
        "U",
        "According to the text, what are the security risks of big data systems?",
        "NOT IN TEXT",
        [
            "Security risks include data breaches, unauthorized access, and compliance violations.",
            "Big data systems face privacy risks due to large-scale data aggregation.",
            "Security concerns arise from storing massive datasets across distributed systems.",
            "Big data introduces cybersecurity risks such as insider threats and external attacks.",
            "Risks include exposure of sensitive information and lack of proper access controls.",
        ],
    ),
    (
        "U2_privacy_laws",
        "U",
        "According to the text, which privacy laws (e.g., GDPR/CCPA) are mentioned as important for big data?",
        "NOT IN TEXT",
        [
            "The text highlights GDPR as a key privacy requirement.",
            "The text mentions GDPR and CCPA as important compliance frameworks.",
            "The text emphasizes HIPAA compliance as a primary privacy concern.",
        ],
    ),
    (
        "U3_mapreduce_definition",
        "U",
        "According to the text, what is MapReduce (give its definition)?",
        "NOT IN TEXT",
        [
            "MapReduce is a programming model for processing large datasets in parallel across a cluster.",
            "MapReduce is a distributed compute framework for batch processing big data.",
            "MapReduce is a database engine optimized for real-time transactions.",
        ],
    ),
    (
        "U4_cost_numbers",
        "U",
        "According to the text, what are the costs (in dollars) of implementing a big data solution?",
        "NOT IN TEXT",
        [
            "Implementing big data typically costs millions of dollars in infrastructure and staffing.",
            "Costs include hardware, software licensing, and operational expenditures.",
            "Organizations should budget for cloud storage and GPU clusters.",
        ],
    ),
    (
        "U5_company_names",
        "U",
        "According to the text, which specific companies are leading big data adoption?",
        "NOT IN TEXT",
        [
            "Companies like Google, Amazon, and Facebook are leading big data adoption.",
            "IBM and Oracle are highlighted as leaders in big data platforms.",
            "Netflix is mentioned as a leader in big data-driven personalization.",
        ],
    ),
]


# ----------------- ONN helpers -----------------
def _tensor_row_to_list(x: Any) -> List[int]:
    if torch.is_tensor(x):
        if x.dim() == 2:
            x = x[0]
        return [int(v) for v in x.flatten().tolist()]
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], list):
            x = x[0]
        return [int(v) for v in x]
    raise TypeError(type(x))


def compress_ids(onn: ONNPreprocessor, ids: List[int]) -> List[int]:
    ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # [1, S]
    out = onn.compress_tokens(ids_t)

    if torch.is_tensor(out) or isinstance(out, list):
        return _tensor_row_to_list(out)
    if isinstance(out, tuple) and len(out) >= 1:
        return _tensor_row_to_list(out[0])
    if isinstance(out, dict):
        for k in ("tokens", "input_ids", "ids", "compressed_tokens", "compressed_ids"):
            if k in out:
                return _tensor_row_to_list(out[k])
    raise TypeError(f"Unexpected compress_tokens return type: {type(out)}")


def lcp_len(a: List[int], b: List[int]) -> int:
    L = min(len(a), len(b))
    i = 0
    while i < L and a[i] == b[i]:
        i += 1
    return i
# ------------------------------------------------


@torch.no_grad()
def score_candidates_batched(
    model,
    tok,
    prompt: str,
    candidates: List[str],
    onn: Optional[ONNPreprocessor],
) -> List[float]:
    """
    Batched candidate scoring: 1 model forward per probe per variant.
    Returns mean logprob per continuation token for each candidate.
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

    full_list: List[List[int]] = []
    prefix_lens: List[int] = []
    true_lens: List[int] = []

    # Build full_ids in *actual* token space used by model
    for cand in candidates:
        cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + cand_ids

        if onn is None:
            full_list.append(full_ids)
            prefix_lens.append(len(prompt_ids))
            true_lens.append(len(full_ids))
        else:
            cp = compress_ids(onn, prompt_ids)
            cf = compress_ids(onn, full_ids)
            pref = lcp_len(cp, cf)
            full_list.append(cf)
            prefix_lens.append(pref)
            true_lens.append(len(cf))

    max_len = max(true_lens)
    batch = []
    attn = []
    for ids, L in zip(full_list, true_lens):
        pad_n = max_len - L
        batch.append(ids + [pad_id] * pad_n)
        attn.append([1] * L + [0] * pad_n)

    input_ids = torch.tensor(batch, dtype=torch.long, device=DEVICE)          # [B, T]
    attention_mask = torch.tensor(attn, dtype=torch.long, device=DEVICE)      # [B, T]

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # [B, T, V]
    logprobs = torch.log_softmax(logits, dim=-1)

    scores: List[float] = []
    for i in range(len(candidates)):
        pref = prefix_lens[i]
        L = true_lens[i]

        if pref <= 0 or pref >= L:
            scores.append(-math.inf)
            continue

        labels = input_ids[i, pref:L]               # [cont_len]
        cont_len = labels.numel()
        if cont_len <= 0:
            scores.append(-math.inf)
            continue

        lp = logprobs[i, (pref - 1):(L - 1), :]     # [cont_len, V]
        if lp.shape[0] != cont_len:
            scores.append(-math.inf)
            continue

        token_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        scores.append(float(token_lp.mean().item()))

    return scores


# ----------------- Variants -----------------
def make_onn_rle(d_model: int) -> ONNPreprocessor:
    return ONNPreprocessor(
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


def make_onn_gated(d_model: int, q: float) -> ONNPreprocessor:
    return ONNPreprocessor(
        ONNConfig(
            d_model=d_model,
            feature_dim=FEATURE_DIM,
            projector_kind="residual_det",
            epsilon=EPSILON,
            seed=SEED,
            compression="onn_gated",
            onn_gate_quantile=float(q),
            onn_gate_min_run=2 if q <= 0.25 else 3,
            onn_gate_threshold=1.4,
        )
    )
# -------------------------------------------


@dataclass
class ProbeOutcome:
    pid: str
    kind: str
    is_correct: bool
    gap: float  # chosen_score - correct_score


def eval_variant(model, tok, onn: Optional[ONNPreprocessor]) -> List[ProbeOutcome]:
    outs: List[ProbeOutcome] = []
    for pid, kind, question, correct, distractors in PROBES:
        prompt = build_prompt(BIGDATA_TEXT, question)
        cands = [correct] + distractors

        scores = score_candidates_batched(model, tok, prompt, cands, onn)
        best_i = max(range(len(cands)), key=lambda i: scores[i])

        correct_score = scores[0]
        chosen_score = scores[best_i]

        outs.append(
            ProbeOutcome(
                pid=pid,
                kind=kind,
                is_correct=(best_i == 0),
                gap=(chosen_score - correct_score),
            )
        )
    return outs


def summarize(outcomes: List[ProbeOutcome]) -> Dict[str, float]:
    A = [o for o in outcomes if o.kind == "A"]
    U = [o for o in outcomes if o.kind == "U"]

    a_acc = sum(o.is_correct for o in A) / max(1, len(A))
    u_acc = sum(o.is_correct for o in U) / max(1, len(U))
    all_acc = sum(o.is_correct for o in outcomes) / max(1, len(outcomes))
    u_gap_mean = sum(o.gap for o in U) / max(1, len(U))  # >0 => hallucinations preferred on avg

    return {"A_acc": a_acc, "U_acc": u_acc, "All_acc": all_acc, "U_gap_mean": u_gap_mean}


def print_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print(f"\nDEVICE: {DEVICE}  DTYPE: {DTYPE}")
    print("RESULTS (candidate scoring; higher is better)")
    print("A_acc = answerable (grounding) accuracy")
    print("U_acc = unanswerable (refusal) accuracy")
    print("U_gap_mean > 0 means hallucinations still preferred on average\n")

    header = f"{'variant':16} | {'A_acc':6} | {'U_acc':6} | {'All':6} | {'U_gap_mean':10}"
    print(header)
    print("-" * len(header))
    for name, s in rows:
        print(
            f"{name:16} | "
            f"{s['A_acc']:.2f}  | "
            f"{s['U_acc']:.2f}  | "
            f"{s['All_acc']:.2f}  | "
            f"{s['U_gap_mean']:+.3f}"
        )


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


def main():
    model, tok = load_model_and_tokenizer()
    d_model = int(getattr(model.config, "hidden_size"))

    rows: List[Tuple[str, Dict[str, float]]] = []

    # baseline
    base = eval_variant(model, tok, None)
    rows.append(("baseline", summarize(base)))

    # rle
    rle = eval_variant(model, tok, make_onn_rle(d_model))
    rows.append(("onn_rle", summarize(rle)))

    # gated sweep
    for q in GATE_QUANTILES:
        name = f"gated_q{q:.2f}"
        outs = eval_variant(model, tok, make_onn_gated(d_model, q))
        rows.append((name, summarize(outs)))

    print_table(rows)


if __name__ == "__main__":
    main()