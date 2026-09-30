# TripSignal AI

## Travel Intelligence Copilot for Planning, Flying, and Recovery

TripSignal AI is a professional RAG-based travel intelligence assistant for flights and trains. It covers the full trip lifecycle: planning a destination, comparing fares, preparing for a long-haul journey, checking live-style flight or train status, planning a long layover, and recovering when a flight is delayed or cancelled.

### Professional framing

Most travel chatbots either answer trivia or only help after something goes wrong. TripSignal is built around the full journey: inspiration, planning, booking tradeoffs, readiness, in-trip status, disruption recovery, and post-disruption benefit checks across air and rail.

### What the MVP does

- Plans destination itineraries with pacing, food, area clustering, and travel-time buffers.
- Compares sample flight deals by total value: price, route, layover, baggage, flexibility, and loyalty or card benefits.
- Builds travel readiness checklists for long-haul trips, documents, baggage, health, devices, and arrival planning.
- Answers flight status questions using structured flight records.
- Handles cancellation recovery: refund path, rebooking choice, vouchers, hotel or meal support, compensation caution, and alternate flights with sample fares.
- Answers train status questions using structured rail records.
- Builds long-layover plans for international itineraries, including lounge access, sleep options, credit-card benefit checks, and return-to-gate buffers.
- Retrieves policy guidance from a local knowledge base using hybrid BM25, TF-IDF, and dense-vector retrieval fused with Reciprocal Rank Fusion, followed by feature reranking.
- Exposes an agent trace for every answer: detected intent, tools used, retrieved documents, LLM composer mode, and verifier result.
- Includes a Self-RAG-lite verifier that checks citation presence and flags prices that are not present in provider inventory.
- Corrective RAG-lite on the policy path: retrieved evidence is graded against the query's concepts (synonym-aware coverage, threshold 0.35); on a low grade the query is rewritten once with policy-vocabulary expansion and retrieval is retried; if the retry still grades low, the system refuses instead of answering. The agent trace exposes the grade, the rewritten query, and the refusal decision.
- Combines live-style status data with RAG so policy answers are grounded in documents.
- Produces action-oriented responses: status, risk, likely impact, recommended next step, and source citations.
- Refuses to invent status when a flight or train is not found.

### Example questions

- Plan a 3 day trip to Dubai.
- Find cheap flights from Washington DC to Hyderabad.
- What should I check before my long-haul trip to India?
- Is AA184 delayed?
- My EK232 flight was cancelled. What refund, voucher, compensation, and alternate flight options do I have?
- My UA920 flight is delayed. Will I miss my connection?
- I am flying from America to India with an 8 hour layover in Dubai. What should I do?
- What happens if Amtrak 391 is late?
- What are my options if my flight is cancelled?
- Is a 45 minute layover in Chicago risky during winter?

### Data sources

TripSignal separates facts into three layers: curated policy documents for RAG, structured provider feeds for live status and prices, and user itinerary context. See [DATA_SOURCES.md](sandbox://workspace/projects/travel-disruption-rag/DATA_SOURCES.md) for the provider plan.

### Architecture

```text
User query
  -> intent and entity extraction (flight number, train number, route, disruption type)
  -> provider adapters for structured status and alternate flight inventory
  -> hybrid BM25 + TF-IDF + dense retrieval over policy and operations documents
  -> evidence grading; on low confidence, one synonym-expansion query rewrite and retry, else refusal
  -> feature reranking, agent tool routing, cited response composition, and verifier checks
  -> cited, action-oriented answer
```

### Production upgrade path

1. Replace sample status JSON with live APIs for flights and trains.
2. Swap the hashing dense baseline for neural embeddings such as BGE-M3 or Qwen3-Embedding in Qdrant.
3. Swap the feature reranker for a cross-encoder such as bge-reranker-v2-m3, Qwen3-Reranker, Cohere Rerank, or Voyage Rerank.
4. Strengthen Corrective RAG beyond the shipped lite version (concept-coverage grading, one rewrite, retry, refusal): learned evidence graders, multiple rewrite rounds, and per-document grading instead of top-1 grading.
5. Add user itinerary memory: home airport, loyalty status, connection airports, notification preferences.
6. Add proactive alerts through email, SMS, or WhatsApp.
7. Expand evaluation: retrieval recall, MRR, citation precision, unsupported-claim rate, delay-answer accuracy, and refusal correctness.

### Run the CLI demo

```bash
cd ~/workspace/projects/travel-disruption-rag
python3 -m app.cli "Is AA184 delayed and what should I do?"
```

### Run the web UI

```bash
cd ~/workspace/projects/travel-disruption-rag
python3 app/server.py 8000
```

Then open `http://127.0.0.1:8000`.

### Run evaluation

```bash
python3 scripts/evaluate.py
```

### Evaluation (measured 2026-09-30, `python3 scripts/evaluate.py`)

Golden set: `data/eval/questions.jsonl` — **53 questions** covering flight delay, cancellation recovery, long layover, train status/delay, exact-entity status, connection risk, compensation, credit-card benefits, passenger rights, trip planning, flight deals, travel readiness, paraphrased/synonym-heavy queries, and refusal cases.

| Benchmark | Result |
|---|---|
| Answer coverage | **53/53 = 100%** |
| Hybrid Mean Recall@3 | **0.977** |
| Hybrid MRR@3 | **0.909** |
| Verifier-supported answers | **53/53 = 100%** |
| Refusal accuracy (refusal-category rows) | **7/7 = 100%** |

Per-arm ablation (Recall@3 / MRR@3 over the 44 gold-bearing rows; refusal and exact-entity rows have no knowledge-base gold documents and are scored by answer/refusal correctness instead):

| Retrieval arm | Recall@3 | MRR@3 |
|---|---|---|
| BM25-only | 0.932 | 0.856 |
| Dense-only (hashing baseline) | 0.727 | 0.712 |
| **Hybrid (BM25 + TF-IDF + dense, RRF + rerank)** | **0.977** | **0.909** |

Honest notes: the hybrid beats each single arm, which is the claim the ablation exists to test. The dense arm is a deterministic hashing baseline (not neural embeddings), and its lower scores show it: on synonym-heavy queries it is the corrective rewrite, not the dense arm, that recovers several answers. Targets from the problem statement (hybrid Recall@3 ≥ 0.90, MRR@3 ≥ 0.85 on a 50+ set) are met on this set. RAGAS-style generation metrics are not claimed; they require the neural/LLM composer path.

### Run tests

```bash
python3 -m pytest tests -q
```

If pytest is not installed, run:

```bash
python3 scripts/run_demo.py
```
