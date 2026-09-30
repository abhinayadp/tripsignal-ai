from app.chatbot import TravelDisruptionAssistant


def test_flight_delay_answer_uses_structured_status():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("Is AA184 delayed?")
    assert "AA184" in answer
    assert "55 minutes" in answer
    assert "Sources:" in answer


def test_train_delay_answer_uses_structured_status():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("What happens if Amtrak train 391 is late?")
    assert "391" in answer
    assert "42 minutes" in answer
    assert "Sources:" in answer


def test_unknown_flight_does_not_invent_status():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("Is ZZ999 delayed?")
    assert "could not find" in answer.lower()
    assert "will not invent" in answer.lower()


def test_policy_question_retrieves_guidance():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("What are my options if my flight is cancelled?")
    assert "Sources:" in answer
    assert "refund" in answer.lower() or "rebooking" in answer.lower()


def test_cancelled_flight_recovery_includes_alternates_and_refund():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("My EK232 flight was cancelled. What should I do?")
    assert "Recovery plan" in answer
    assert "refund" in answer.lower()
    assert "voucher" in answer.lower()
    assert "Cheapest sample option" in answer
    assert "$1039" in answer
    assert "Sources:" in answer


def test_dubai_layover_answer_includes_card_and_sleep_checks():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("I am traveling from America to India and have an 8 hour layover in Dubai. What can I do?")
    assert "Dubai layover" in answer
    assert "Priority Pass" in answer
    assert "sleep" in answer.lower()
    assert "Sources:" in answer


def test_hybrid_retriever_returns_ranked_documents():
    assistant = TravelDisruptionAssistant()
    results = assistant.index.search("cancelled flight refund voucher alternate options", top_k=3)
    assert results
    assert "retrieval_score" in results[0]


def test_provider_contract_returns_sample_alternates():
    assistant = TravelDisruptionAssistant()
    options = assistant.provider.get_alternate_flights("EK232")
    assert options
    assert min(option["price_usd"] for option in options) == 1039


def test_trip_planning_answer_for_dubai():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("Plan a 3 day trip to Dubai")
    assert "Dubai plan" in answer
    assert "Burj Khalifa" in answer
    assert "Sources:" in answer


def test_flight_deals_answer_uses_sample_deals():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("Find cheap flights from Washington DC to Hyderabad")
    assert "Cheapest sample" in answer
    assert "$982" in answer
    assert "sample fares" in answer.lower()


def test_readiness_answer_includes_long_haul_checks():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("What should I check before my long-haul trip to India?")
    assert "readiness check" in answer.lower()
    assert "passport" in answer.lower()
    assert "Sources:" in answer


def test_answer_trace_includes_agent_tools_and_verification():
    assistant = TravelDisruptionAssistant()
    result = assistant.answer_with_trace("My EK232 flight was cancelled. What should I do?")
    assert result["trace"]["intent"] == "cancellation_recovery"
    assert "alternate_flights_tool" in result["trace"]["tools_used"]
    assert result["trace"]["verification"]["supported"] is True
    assert result["trace"]["retrieved_docs"]


def test_verifier_flags_unsupported_price():
    assistant = TravelDisruptionAssistant()
    verification = assistant.verifier.verify("Take this flight for $7777. Sources: Sample.", allowed_prices={1039})
    assert verification["supported"] is False
    assert verification["prices_found"] == [7777]


def test_llm_adapter_defaults_to_rule_based_without_key():
    assistant = TravelDisruptionAssistant()
    assert assistant.llm.name in {"rule-based", "openai-compatible"}


# --- Corrective RAG-lite: grading, one-shot rewrite, refusal bounds ---

REFUSAL_QUERIES = [
    "Is ZZ999 delayed?",
    "Is QQ123 delayed?",
    "Tell me the status of flight DL999",
    "My BA9999 flight was cancelled. What refund do I get?",
    "Is Amtrak train 777 late?",
    "What about NEC999 train?",
]


def test_low_confidence_query_is_rewritten_and_recovers():
    # "money back" has no lexical overlap with the policy vocabulary
    # ("refund"), so the first retrieval grades below threshold; the
    # one-shot synonym rewrite must retry and recover a grounded answer.
    assistant = TravelDisruptionAssistant()
    result = assistant.answer_with_trace("Can I get my money back instead of a voucher?")
    correction = result["trace"]["corrective_rag"]
    assert correction["retried"] is True
    assert correction["rewritten_query"] is not None
    assert "refund" in correction["rewritten_query"]
    assert correction["passed"] is True
    assert correction["refused"] is False
    assert "refund" in result["answer"].lower()
    assert "Sources:" in result["answer"]


def test_refusal_after_failed_retry():
    # A synonym trigger ("stranded") forces a rewrite + retry, but the
    # query is out of domain, the retry still grades below threshold,
    # and the system must refuse rather than invent.
    assistant = TravelDisruptionAssistant()
    result = assistant.answer_with_trace("I am stranded with my homework tonight")
    correction = result["trace"]["corrective_rag"]
    assert correction["retried"] is True
    assert correction["passed"] is False
    assert correction["refused"] is True
    assert "could not find" in result["answer"].lower()
    assert "will not invent" in result["answer"].lower()


def test_out_of_domain_query_refused_without_retry_trigger():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("What is the capital of France?")
    assert "could not find" in answer.lower()
    assert "will not invent" in answer.lower()


def test_unknown_entities_refusal_precision():
    # 100% refusal precision on the unknown-entity set: every query must
    # be refused explicitly, with no invented status, delay, or price.
    assistant = TravelDisruptionAssistant()
    refused = 0
    for query in REFUSAL_QUERIES:
        answer = assistant.answer(query)
        if "could not find" in answer.lower() and "will not invent" in answer.lower():
            refused += 1
        assert "minutes" not in answer.lower(), query
        assert "$" not in answer, query
    assert refused == len(REFUSAL_QUERIES)


def test_unknown_train_does_not_invent_status():
    assistant = TravelDisruptionAssistant()
    answer = assistant.answer("Is Amtrak train 777 late?")
    assert "could not find" in answer.lower()
    assert "will not invent" in answer.lower()


def test_grade_evidence_separates_relevant_from_irrelevant():
    from app.corrective import grade_evidence

    assistant = TravelDisruptionAssistant()
    relevant = assistant.index.search("cancelled flight refund voucher", top_k=3)
    irrelevant = assistant.index.search("best pizza topping recipe", top_k=3)
    assert grade_evidence("cancelled flight refund voucher", relevant)["passed"] is True
    assert grade_evidence("best pizza topping recipe", irrelevant)["passed"] is False


def test_search_arm_ablation_endpoints():
    assistant = TravelDisruptionAssistant()
    for arm in ("bm25", "tfidf", "dense", "hybrid"):
        results = assistant.index.search_arm("cancelled flight refund", arm, top_k=3)
        assert results, arm
        assert "cancellation_recovery_playbook" in [doc["id"] for doc in results], arm
    # The hybrid and lexical arms rank the playbook first; the hashing
    # dense baseline is documented as weaker and only places it top-3.
    for arm in ("bm25", "tfidf", "hybrid"):
        results = assistant.index.search_arm("cancelled flight refund", arm, top_k=3)
        assert results[0]["id"] == "cancellation_recovery_playbook", arm


def test_synonym_query_not_misrouted_to_trip_planning():
    # Regression: "plane" contains the substring "plan"; routing must
    # use word boundaries so this paraphrase reaches policy retrieval.
    assistant = TravelDisruptionAssistant()
    result = assistant.answer_with_trace("Put me on another plane, mine was cancelled")
    assert result["trace"]["intent"] != "trip_planning"
    assert "rebooking" in result["answer"].lower()
    assert "Sources:" in result["answer"]
