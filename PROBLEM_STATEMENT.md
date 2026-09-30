# TripSignal AI — Problem Statement

## A Semantic-Search-First, Hybrid RAG Travel Intelligence Copilot

> One-line version: When a flight is cancelled at midnight, a traveller does not need a chatbot. They need the *right clause, from the right policy, fused with their live flight status* — in seconds, with citations, and with a system that refuses rather than invents when it does not know.

---

## 1. Domain Asymmetry & Information Bottleneck

**The asymmetry:** Airlines, rail operators, airports, and card issuers hold the ground truth. The traveller holds a boarding pass and a deadline.

**Where the ground truth lives:** It is fragmented across thousands of heterogeneous, version-volatile sources — airline Conditions of Carriage, DOT / EU261 / India DGCA passenger-rights rules, refund and rebooking policies, fare rules, lounge and credit-card benefit terms, airport and layover guides, and carrier operations manuals. No single document answers a real question.

**Cognitive load & latency:** A real disruption question is never a lookup, it is a cross-reference:

> *"My EK232 was cancelled — am I owed a refund or just a voucher, can I take the alternate flight tomorrow, and does my card cover a hotel tonight?"*

Answering that correctly requires joining the cancellation playbook + passenger rights + card benefit terms + live alternate-flight inventory. A stressed traveller at an airport cannot do that with Ctrl+F: the vocabulary does not match theirs ("involuntary refund", "Schedule change", "Rule 240"), the clauses are dense, and the clock is running. This is exactly why **semantic search is a core component, not a feature** — the user's words and the policy's words are rarely the same words.

**Cost of inaction / delay:** A missed rebooking window (seats on alternates sell out in minutes), a voucher accepted instead of a refund owed, an expired compensation claim, a missed connection, or a night paid out-of-pocket that a benefit would have covered.

## 2. Multi-Modal Data Friction

Basic chunk-and-embed fails here for specific, structural reasons:

- **Structural complexity:** Ground truth is trapped in fare tables, eligibility matrices, multi-row compensation tables (delay length × distance × amount), footnotes, and key-value itinerary schemas — not clean prose. Naive chunking severs a table row from its header and a rule from its exception.
- **Temporal drift / versioning:** Policies, fare rules, and schedules change. A 2024 compensation chunk and a 2026 chunk can both retrieve highly and directly contradict each other. Retrieval must be version-aware, not just similarity-aware.
- **Dual-state data:** A correct answer must fuse *unstructured* text (policies, playbooks, guides) with *structured operational state* (live flight/train status, delay minutes, alternate-flight price and timing from provider APIs / SQL records). Text-only RAG cannot know AA184 is delayed 55 minutes; status-only APIs cannot explain what that delay entitles you to.

TripSignal therefore separates three layers: curated policy documents for RAG, structured provider feeds for status/prices, and user itinerary context — and composes them in one answer.

## 3. Technical Failure Modes of "Naive RAG"

TripSignal exists because a one-pass vector search fails in four documented ways:

1. **Lost in the Middle & Context Contamination:** Packing top-K chunks into a prompt dilutes the one clause that matters. More context lowers generator accuracy when most of it is irrelevant. Retrieval precision matters more than retrieval volume.
2. **Dense Embedding Blindspots (why semantic search alone is not enough):** Pure semantic / bi-encoder search is strong on intent and paraphrase — *"flight got scrapped, what can I get back?"* → cancellation/refund policy — but weak on exact alphanumeric entities: `EK232`, `AA184`, `Amtrak 391`, airport codes, and clause numbers. Those must match exactly.
3. **Lexical Blindspots (why BM25 alone is not enough):** BM25/TF-IDF fails on synonyms and rephrasing. Users say "stranded overnight", "put me on another plane", "is this layover too tight?" — documents say "involuntary rebooking", "misconnection risk", "minimum connect time". Only a dense semantic arm bridges that gap.
4. **The Static Retrieval Trap:** A standard top-K pipeline returns *something* whether or not it is relevant. It has no grading, no query rewrite, and no refusal — so for an unindexed flight (`ZZ999`) it will confidently ground an answer in the nearest, wrong document.

## 4. Risk Vector & Hallucination Cost

- **Factual hallucination:** Inventing a delay time, a fare ($ price not in inventory), a compensation amount, or a departure time. In travel, a fabricated number is not a style error — the traveller acts on it.
- **Semantic hallucination:** Stating a plausible-sounding entitlement that does not exist in the retrieved policy ("you are guaranteed a hotel") when the policy says discretionary.
- **Liability & trust cost:** Wrong refund/compensation guidance = direct financial loss and a forfeited claim. A wrong "you will make your connection" = a stranded passenger. Therefore the system's correct behaviour on insufficient evidence is an **explicit refusal** ("could not find / will not invent"), not a best guess. A refusal preserves trust; a confident invention destroys it in one answer.

## 5. Architectural Objectives & Constraints

**Objective:** Retrieve with semantic understanding, verify with lexical precision, answer only with cited evidence.

1. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF) — implemented**
   - **Dense semantic arm:** neural embeddings (production: BGE-M3 / BGE-small via the `SentenceTransformerEmbeddingProvider` adapter, in Qdrant at scale) for intent, synonyms, and paraphrase.
   - **Sparse arms:** BM25 + TF-IDF for exact flight/train numbers, codes, and policy terminology.
   - RRF (k=60) fuses the ranked lists without score normalisation, so no single arm can dominate.
   - *Honest current state:* the shipped dense arm is a deterministic hashing baseline behind the same provider contract, so the neural swap changes the provider, not the retriever. Claiming semantic search without wiring the neural provider would be an overclaim — it is the named production upgrade.
2. **Reranking — implemented as feature reranking, cross-encoder as upgrade**
   Candidate pool (top-20 per arm) is reranked on query coverage and title alignment before generation. Production target: cross-encoder reranker (bge-reranker-v2-m3 / Qwen3-Reranker) to maximise signal-to-noise in the context window.
3. **Self-Correction & Refusal Bounds — implemented (Self-RAG-lite + Corrective RAG-lite)**
   A verifier checks citation presence and flags any price not present in provider inventory. Unknown flight/train entities trigger explicit refusal. Corrective RAG-lite is implemented on the policy path: retrieved evidence is graded by synonym-aware concept coverage (threshold 0.35), a low grade triggers one policy-vocabulary query rewrite and a retrieval retry, and a still-low grade triggers refusal. Production target: full Corrective RAG with learned per-document graders and multiple rewrite rounds.
4. **System constraints (targets)**
   - p95 latency < 2.5s for a grounded answer
   - Deterministic citation attribution on every factual claim ("Sources:")
   - Fixed token budget: top-3 reranked chunks only — precision over packing
   - Full agent trace per answer: intent, tools used, retrieved docs, composer mode, verifier result

## 6. Concrete, Measurable Evaluation Benchmarks

Success is proven, not asserted. Golden set: `data/eval/questions.jsonl` — **53 questions** across flight delay, cancellation recovery, long layover, train status/delay, exact-entity status, connection risk, compensation, credit-card benefits, passenger rights, trip planning, flight deals, travel readiness, paraphrased/synonym-heavy queries, and refusal cases.

**Measured (verified 2026-09-30, `python3 scripts/evaluate.py`):**

| Benchmark | Result |
|---|---|
| Answer coverage | **53/53 = 100%** |
| Hybrid Mean Recall@3 | **0.977** |
| Hybrid MRR@3 | **0.909** |
| Verifier-supported answers | **53/53 = 100%** |
| Refusal accuracy (refusal-category rows) | **7/7 = 100%** |

Retrieval metrics are computed over the 44 gold-bearing rows; refusal and exact-entity rows have no knowledge-base gold documents (status comes from structured provider data, not retrieval) and are scored by answer/refusal correctness instead.

**Per-arm ablation (gold-bearing rows):**

| Retrieval arm | Recall@3 | MRR@3 |
|---|---|---|
| BM25-only | 0.932 | 0.856 |
| Dense-only (hashing baseline) | 0.727 | 0.712 |
| **Hybrid (BM25 + TF-IDF + dense, RRF + rerank)** | **0.977** | **0.909** |

**Targets vs measured:**

- **Retrieval:** Recall@3 ≥ 0.90 and MRR@3 ≥ 0.85 on the expanded 50+ golden set — **met** (0.977 / 0.909, hybrid). Per-arm reporting shows the hybrid beats each single arm. Honest caveat: the dense arm is a hashing baseline, not neural embeddings, and it is the weakest arm (0.727 / 0.712); on synonym-heavy queries, the corrective rewrite recovers several answers the dense arm alone misses. The semantic-arm claim therefore stays bounded: fusion + correction are proven on this set; a neural dense arm remains the named production upgrade.
- **Generation (RAGAS):** Faithfulness / Groundedness ≥ 0.90, Context Precision ≥ 0.85, Context Recall ≥ 0.85, Answer Relevancy ≥ 0.85 — **not yet claimed**; these require the LLM composer path and a RAGAS harness, neither of which is wired in v1.
- **Safety:** 100% precision on out-of-domain / unknown-entity refusal testing (e.g. `ZZ999`) — **met on this set**: refusal-category rows 7/7 refused, and a separate unknown-entity unit-test set (ZZ999, QQ123, DL999, BA9999, Amtrak 777, NEC999) refuses 6/6 with zero invented statuses or prices.
- **Ablation honesty:** any architecture claim (RRF, reranking, semantic arm) must survive an ablation: remove it, show the metric drop, or remove the claim. The per-arm table above is that ablation for retrieval arms; reranker and fusion ablations remain future work and no separate claim is made for them.

---

### Why this is not a "typical RAG demo"

A typical demo embeds clean text, does one vector lookup, and hopes. TripSignal's problem is harder in every dimension that matters in production: fragmented and versioned ground truth, tables and structured state fused with prose, exact-entity + semantic-intent retrieval at the same time, and a domain where a hallucinated number costs the user real money. The architecture, the refusal behaviour, and the evaluation harness are all built — and measured — against that problem.
