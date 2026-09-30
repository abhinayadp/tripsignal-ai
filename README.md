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
  -> feature reranking, agent tool routing, cited response composition, and verifier checks
  -> cited, action-oriented answer
```

### Production upgrade path

1. Replace sample status JSON with live APIs for flights and trains.
2. Swap the hashing dense baseline for neural embeddings such as BGE-M3 or Qwen3-Embedding in Qdrant.
3. Swap the feature reranker for a cross-encoder such as bge-reranker-v2-m3, Qwen3-Reranker, Cohere Rerank, or Voyage Rerank.
4. Strengthen Corrective RAG: grade retrieved evidence, rewrite weak queries, retry retrieval, and refuse when evidence is insufficient.
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

### Run tests

```bash
python3 -m pytest tests -q
```

If pytest is not installed, run:

```bash
python3 scripts/run_demo.py
```
