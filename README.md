# TripSignal AI

## Travel Disruption Intelligence Copilot

TripSignal AI is a professional RAG-based travel operations assistant for flights and trains. Instead of acting like a generic travel chatbot, it is designed as a disruption intelligence system: it checks live-style status records, retrieves carrier policies and passenger-rights guidance, explains why a trip is at risk, and recommends the next best action.

### Professional framing

Most travel chatbots answer trivia. TripSignal is built around a real operational problem: travelers and support teams need one reliable place to understand delays, missed connections, rebooking options, refund rules, and escalation paths across air and rail.

### What the MVP does

- Answers flight status questions using structured flight records.
- Handles cancellation recovery: refund path, rebooking choice, vouchers, hotel or meal support, compensation caution, and alternate flights with sample fares.
- Answers train status questions using structured rail records.
- Builds long-layover plans for international itineraries, including lounge access, sleep options, credit-card benefit checks, and return-to-gate buffers.
- Retrieves policy guidance from a local knowledge base using hybrid BM25 plus TF-IDF ranking fused with Reciprocal Rank Fusion.
- Combines live-style status data with RAG so policy answers are grounded in documents.
- Produces action-oriented responses: status, risk, likely impact, recommended next step, and source citations.
- Refuses to invent status when a flight or train is not found.

### Example questions

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
  -> hybrid BM25 + TF-IDF retrieval over policy and operations documents
  -> response composer combining status + retrieved policy evidence
  -> cited, action-oriented answer
```

### Production upgrade path

1. Replace sample status JSON with live APIs for flights and trains.
2. Swap BM25-only retrieval for hybrid BM25 + dense embeddings + reranking.
3. Add Corrective RAG: grade retrieved evidence, rewrite weak queries, and refuse when evidence is insufficient.
4. Add user itinerary memory: home airport, loyalty status, connection airports, notification preferences.
5. Add proactive alerts through email, SMS, or WhatsApp.
6. Add evaluation: retrieval recall, citation precision, delay-answer accuracy, and refusal correctness.

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
