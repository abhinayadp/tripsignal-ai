# TripSignal Data Sources

TripSignal should never ask the language model to invent travel facts. The model receives data from three controlled layers: a curated knowledge base for RAG, structured provider feeds for status and prices, and user-provided itinerary context.

## 1. Policy and guidance knowledge base for RAG

Used for explanations, refund guidance, passenger rights, disruption playbooks, layover planning, and benefit checks.

Sources to curate:
- U.S. Department of Transportation airline cancellation, refund, tarmac delay, and customer service pages.
- Airline customer service plans and contract-of-carriage pages for major carriers used in the app.
- Amtrak guest relations, fare rules, station information, and service disruption notices.
- Airport official pages for layover services, lounges, transit hotels, sleep pods, baggage storage, and terminal maps.
- Credit card benefit guides from issuers, stored as card-product-specific documents and always shown with a verify-your-card caveat.
- Loyalty program pages for airline status, alliance lounge access, and mileage or voucher rules.

Pipeline:
Official page or PDF -> parser -> clean text -> metadata such as carrier, airport, topic, effective date, and source URL -> chunking -> embedding and BM25 index -> cited retrieval.

## 2. Live flight status

Used when the user enters a flight number or clicks a flight card.

Provider options:
- FlightAware AeroAPI for flight status, estimated times, gates, terminals, and disruption details. Usually paid.
- Cirium or FlightStats class providers for enterprise-grade operational status. Usually paid.
- AviationStack or similar APIs for a lower-cost prototype feed. Quality and coverage vary by carrier.
- OpenSky Network for aircraft position and track data. Useful for map context, not a complete source for gates, cancellation, or passenger recovery options.

Normalized fields:
flight number, carrier, origin, destination, scheduled and estimated departure, scheduled and estimated arrival, status, delay minutes, gate, terminal, cause if provided, source provider, and fetched timestamp.

## 3. Alternate flights, schedules, and prices

Used for cancellation recovery and rebooking recommendations.

Provider options:
- Amadeus Self-Service APIs for flight offers, schedules, prices, and availability in a developer environment.
- Duffel for flight search and booking-oriented workflows.
- Skyscanner or Kiwi style partner APIs where access is approved.
- Direct airline APIs or NDC connections for production deployments.

Important rule:
Prices and seats are volatile. TripSignal must show the provider, search timestamp, cabin, baggage caveat, and whether the fare is bookable or only a sample estimate. In the MVP, alternate fares are sample inventory and must be labeled as sample data.

## 4. Train status and schedules

Used when the user enters an Amtrak or regional rail train number.

Provider options:
- Amtrak GTFS static schedule data for planned routes and stops.
- Agency GTFS-Realtime feeds where available for regional rail alerts, trip updates, and vehicle positions.
- Amtrak status pages or partner feeds where licensing allows.
- Transit agency APIs for local commuter rail systems.

Important rule:
U.S. long-distance rail real-time data is less standardized than flight data. TripSignal should show the last update timestamp and next decision point instead of pretending the estimate is exact.

## 5. Airports, airlines, and reference data

Used to resolve codes and enrich answers.

Sources:
- OurAirports and OpenFlights style datasets for airport codes, cities, time zones, and coordinates.
- Airline and alliance reference tables maintained in the project.
- Airport official websites for terminal and layover services.

## 6. Weather and disruption context

Used to explain risk, not to overstate causality.

Sources:
- NOAA and National Weather Service for U.S. weather alerts.
- METAR and TAF aviation weather feeds for airport conditions.
- OpenWeather or similar providers for simple forecast context.
- FAA airport status and delay programs where available.

TripSignal should phrase weather as context: "weather is affecting the hub" rather than claiming it is the confirmed cause unless the carrier or provider states the cause.

## 7. User itinerary and card context

Used only with user permission.

Inputs:
- Flight or train number.
- Booking route and dates pasted by the user.
- Airline status level.
- Credit card product name, never card number.
- Home airport, preferred airports, notification channel.

Stored separately from the public knowledge base so personal itinerary data is never mixed into general retrieval.

## Recommended production architecture

Provider APIs and official documents -> ingestion jobs -> normalized Postgres tables for status, offers, airports, and itineraries -> Qdrant or similar vector index for policy documents -> FastAPI tools for status lookup, alternate flight search, benefit checks, and RAG retrieval -> LLM response composer with citations and timestamps.
