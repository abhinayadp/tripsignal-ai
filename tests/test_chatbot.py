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
