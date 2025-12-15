from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from onnpp.api.preprocessor import ONNPreprocessor, ONNConfig


# ----------------- Hardware / dtype -----------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
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


GATE_QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Single canonical refusal used as the ONLY correct answer for ALL unanswerables
CANON_REFUSAL = "NOT IN TEXT."


PROMPT_TEMPLATES: List[str] = [
    # Pass 1: harsh stress test
    (
        "You are given a source text. Answer the question using ONLY the source text.\n"
        "If the answer is not in the text, respond exactly with: NOT IN TEXT.\n"
        "Do not add any extra words.\n\n"
        "SOURCE TEXT:\n{doc}\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    ),
    # Pass 2: policy pass (ANCHOR to the canonical refusal string)
    (
        "Use ONLY the source text to answer.\n"
        f"If the information is missing, reply exactly with: {CANON_REFUSAL}\n"
        "Do not guess.\n\n"
        "SOURCE TEXT:\n{doc}\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    ),
    # Pass 3: another stress framing
    (
        "Answer strictly from SOURCE TEXT.\n"
        "If unsupported, you MUST say: NOT IN TEXT.\n"
        "Be concise.\n\n"
        "[SOURCE]\n{doc}\n\n"
        "[QUESTION]\n{question}\n\n"
        "[ANSWER]\n"
    ),
]


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


def build_prompt(template: str, doc: str, question: str) -> str:
    return template.format(doc=doc, question=question)


# probe_id, kind, question, correct, distractors
# kind: "A" = answerable, "U" = unanswerable
#
# v4 changes:
# - ALL U probes correct answer is exactly CANON_REFUSAL.
# - “Explanatory refusals” are distractors (to measure strict refusal preference).
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
        CANON_REFUSAL,
        [
            # explanatory refusals (should still count as wrong in strict mode)
            "NOT IN TEXT. The source does not mention security risks.",
            "NOT IN TEXT. The source does not mention this.",
            # direct hallucinations
            "Security risks include data breaches, unauthorized access, and compliance violations.",
            "Big data introduces cybersecurity risks such as insider threats and external attacks.",
            # hedged hallucinations
            "The source does not explicitly discuss security risks, but generally these include breaches, privacy loss, and unauthorized access.",
        ],
    ),
    (
        "U2_privacy_laws",
        "U",
        "According to the text, which privacy laws (e.g., GDPR/CCPA) are mentioned as important for big data?",
        CANON_REFUSAL,
        [
            "NOT IN TEXT. The source does not mention any privacy laws.",
            "NOT IN TEXT. The source does not mention this.",
            "The text mentions GDPR and CCPA as important compliance frameworks.",
            "The text highlights GDPR as a key privacy requirement.",
            "The source does not list laws, but GDPR/CCPA are generally important for big data compliance.",
        ],
    ),
    (
        "U3_mapreduce_definition",
        "U",
        "According to the text, what is MapReduce (give its definition)?",
        CANON_REFUSAL,
        [
            "NOT IN TEXT. The source mentions MapReduce but does not define it.",
            "NOT IN TEXT. The source does not define MapReduce.",
            "MapReduce is a programming model for processing large datasets in parallel across a cluster.",
            "MapReduce is a distributed compute framework for batch processing big data.",
            "The text does not define MapReduce, but generally it is a distributed processing framework for large datasets.",
        ],
    ),
    (
        "U4_cost_numbers",
        "U",
        "According to the text, what are the costs (in dollars) of implementing a big data solution?",
        CANON_REFUSAL,
        [
            "NOT IN TEXT. The source does not provide cost numbers.",
            "NOT IN TEXT. The source does not mention this.",
            "Implementing big data typically costs millions of dollars in infrastructure and staffing.",
            "Costs include hardware, software licensing, and operational expenditures.",
            "The text does not give costs, but typically costs are high due to infrastructure and staffing.",
        ],
    ),
    (
        "U5_company_names",
        "U",
        "According to the text, which specific companies are leading big data adoption?",
        CANON_REFUSAL,
        [
            "NOT IN TEXT. The source does not name any companies.",
            "NOT IN TEXT. The source does not mention this.",
            "Companies like Google, Amazon, and Facebook are leading big data adoption.",
            "IBM and Oracle are highlighted as leaders in big data platforms.",
            "The text does not name companies, but major cloud providers typically lead adoption.",
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
    ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
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
def score_candidates_batched(model, tok, prompt: str, candidates: List[str], onn: Optional[ONNPreprocessor]) -> List[float]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]

    full_list: List[List[int]] = []
    prefix_lens: List[int] = []
    true_lens: List[int] = []

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

    input_ids = torch.tensor(batch, dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=DEVICE)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = torch.log_softmax(logits, dim=-1)

    scores: List[float] = []
    for i in range(len(candidates)):
        pref = prefix_lens[i]
        L = true_lens[i]
        if pref <= 0 or pref >= L:
            scores.append(-math.inf)
            continue

        labels = input_ids[i, pref:L]
        cont_len = labels.numel()
        if cont_len <= 0:
            scores.append(-math.inf)
            continue

        lp = logprobs[i, (pref - 1):(L - 1), :]
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
class AggStats:
    A_total: int = 0
    A_ok: int = 0
    U_total: int = 0
    U_ok: int = 0
    U_gap_sum: float = 0.0


def eval_variant_pass(model, tok, onn: Optional[ONNPreprocessor], pass_idx: int) -> AggStats:
    agg = AggStats()
    tmpl = PROMPT_TEMPLATES[pass_idx]

    for pid, kind, question, correct, distractors in PROBES:
        prompt = build_prompt(tmpl, BIGDATA_TEXT, question)
        cands = [correct] + distractors
        scores = score_candidates_batched(model, tok, prompt, cands, onn)
        best_i = max(range(len(cands)), key=lambda i: scores[i])

        correct_score = scores[0]
        chosen_score = scores[best_i]
        is_ok = (best_i == 0)

        if kind == "A":
            agg.A_total += 1
            agg.A_ok += int(is_ok)
        else:
            agg.U_total += 1
            agg.U_ok += int(is_ok)
            agg.U_gap_sum += (chosen_score - correct_score)

    return agg


def eval_variant_allpasses(model, tok, onn: Optional[ONNPreprocessor]) -> AggStats:
    agg = AggStats()
    for pidx in range(len(PROMPT_TEMPLATES)):
        s = eval_variant_pass(model, tok, onn, pidx)
        agg.A_total += s.A_total
        agg.A_ok += s.A_ok
        agg.U_total += s.U_total
        agg.U_ok += s.U_ok
        agg.U_gap_sum += s.U_gap_sum
    return agg


def stats_to_metrics(s: AggStats) -> Tuple[float, float, float, float]:
    A_acc = s.A_ok / s.A_total if s.A_total else 0.0
    U_acc = s.U_ok / s.U_total if s.U_total else 0.0
    All_acc = (s.A_ok + s.U_ok) / (s.A_total + s.U_total) if (s.A_total + s.U_total) else 0.0
    U_gap_mean = s.U_gap_sum / s.U_total if s.U_total else 0.0
    return A_acc, U_acc, All_acc, U_gap_mean


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


def print_table(title: str, rows: List[Tuple[str, float, float, float, float]]) -> None:
    print(f"\n{title}")
    print("A_acc = answerable (grounding) accuracy")
    print("U_acc = unanswerable (refusal) accuracy")
    print("U_gap_mean > 0 means hallucinations still preferred on average\n")

    header = f"{'variant':16} | {'A_acc':6} | {'U_acc':6} | {'All':6} | {'U_gap_mean':10}"
    print(header)
    print("-" * len(header))
    for name, a_acc, u_acc, all_acc, u_gap in rows:
        print(f"{name:16} | {a_acc:.2f}  | {u_acc:.2f}  | {all_acc:.2f}  | {u_gap:+.3f}")


def debug_pass2_winners(model, tok, variants, variant_names):
    print("\n=== PASS 2 PER-PROBE WINNERS (UNANSWERABLE ONLY) ===\n")
    tmpl = PROMPT_TEMPLATES[1]  # Pass 2

    for pid, kind, question, correct, distractors in PROBES:
        if kind != "U":
            continue

        prompt = build_prompt(tmpl, BIGDATA_TEXT, question)
        candidates = [correct] + distractors

        print(f"[{pid}] {question}")

        for name, onn in zip(variant_names, variants):
            scores = score_candidates_batched(model, tok, prompt, candidates, onn)
            best_i = max(range(len(candidates)), key=lambda i: scores[i])
            chosen = candidates[best_i]
            marker = "✓" if best_i == 0 else "✗"
            print(f"  {name:12} → {marker}  {chosen}")

        print("")


def main():
    model, tok = load_model_and_tokenizer()
    d_model = int(getattr(model.config, "hidden_size"))

    print(f"DEVICE: {DEVICE}  DTYPE: {DTYPE}")
    print(f"Canonical refusal (correct for all U probes): {CANON_REFUSAL}")

    variants: List[Tuple[str, Optional[ONNPreprocessor]]] = []
    variants.append(("baseline", None))
    variants.append(("onn_rle", make_onn_rle(d_model)))
    for q in GATE_QUANTILES:
        variants.append((f"gated_q{q:.2f}", make_onn_gated(d_model, q)))

    # Pass 2 only (policy pass)
    pass2_rows: List[Tuple[str, float, float, float, float]] = []
    for name, onn in variants:
        s2 = eval_variant_pass(model, tok, onn, pass_idx=1)
        pass2_rows.append((name, *stats_to_metrics(s2)))
    print_table("PASS 2 ONLY (recommended policy pass)", pass2_rows)

    # Aggregate across all passes
    all_rows: List[Tuple[str, float, float, float, float]] = []
    for name, onn in variants:
        sall = eval_variant_allpasses(model, tok, onn)
        all_rows.append((name, *stats_to_metrics(sall)))
    print_table("AGGREGATE ACROSS ALL PASSES (stress test)", all_rows)

    # Debug for baseline vs two gated settings
    debug_variants = [
        None,
        make_onn_gated(d_model, 0.15),
        make_onn_gated(d_model, 0.35),
    ]
    debug_names = ["baseline", "gated_q0.15", "gated_q0.35"]
    debug_pass2_winners(model, tok, debug_variants, debug_names)


if __name__ == "__main__":
    main()