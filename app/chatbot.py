import json
import re
from pathlib import Path

from .providers import SampleTravelDataProvider
from .retriever import HybridRetriever

BASE_DIR = Path(__file__).resolve().parents[1]
KB_DIR = BASE_DIR / "data" / "knowledge_base"
STATUS_DIR = BASE_DIR / "data" / "status"

FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{2,4})\b")
TRAIN_RE = re.compile(r"\b(?:amtrak\s*)?(\d{2,3}|NEC\d{3})\b", re.IGNORECASE)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = text.splitlines()[0].replace("#", "").strip() if text else path.stem
        docs.append({"id": path.stem, "title": title, "path": str(path), "text": text})
    return docs


class TravelDisruptionAssistant:
    """RAG assistant that separates structured status lookup from policy retrieval."""

    def __init__(self, provider: SampleTravelDataProvider | None = None):
        self.documents = load_documents()
        self.index = HybridRetriever(self.documents)
        self.provider = provider or SampleTravelDataProvider()
        self.flights = self.provider.flights
        self.trains = self.provider.trains
        self.alternates = self.provider.alternates
        self.deals = self.provider.deals

    def _flight_answer(self, flight_number: str, query: str) -> str | None:
        record = self.flights.get(flight_number)
        if not record:
            return (
                f"I could not find structured status for {flight_number}. "
                "I will not invent a delay estimate; please verify the flight number or carrier."
            )

        evidence = self.index.search(query + " flight delayed connection rebooking", top_k=2)
        citations = ", ".join(doc["title"] for doc in evidence) or "Airline Delay and Cancellation Guidance"
        return (
            f"{record['carrier']} flight {record['flight_number']} from {record['origin']} to {record['destination']} "
            f"is {record['status'].lower()}. Scheduled departure was {record['scheduled_departure']}; "
            f"current estimate is {record['estimated_departure']} from gate {record['gate']}. "
            f"Delay: {record['delay_minutes']} minutes. Cause: {record['cause']}. "
            f"Connection risk: {record['connection_risk']}. Recommended action: "
            f"{self._recommended_action(record['delay_minutes'], record['connection_risk'])} "
            f"Sources: {citations}."
        )

    def _train_answer(self, train_key: str, query: str) -> str | None:
        record = self.trains.get(train_key) or self.trains.get(train_key.upper())
        if not record:
            return None

        evidence = self.index.search(query + " train delayed station connection arrival", top_k=2)
        citations = ", ".join(doc["title"] for doc in evidence) or "Amtrak and Rail Delay Guidance"
        return (
            f"{record['operator']} train {record['train_number']} ({record['train_name']}) from {record['origin']} "
            f"to {record['destination']} is {record['status'].lower()}. Delay: {record['delay_minutes']} minutes. "
            f"Cause: {record['cause']}. Current arrival estimate is {record['estimated_arrival']}. "
            f"Next decision point: {record['next_decision_point']}. Recommended action: "
            f"{self._train_action(record['delay_minutes'])} Sources: {citations}."
        )

    @staticmethod
    def _recommended_action(delay_minutes: int, risk: str) -> str:
        if risk == "High" or delay_minutes >= 90:
            return "contact the carrier now, ask for a protected backup or rebooking, and do not wait until landing to fix the connection."
        if delay_minutes >= 30:
            return "stay near the gate, monitor updates every 15 minutes, and prepare a backup connection option."
        if delay_minutes > 0:
            return "stay close to the departure area and re-check before leaving for food or another terminal."
        return "proceed normally, but keep notifications on because gate and departure estimates can change."

    @staticmethod
    def _train_action(delay_minutes: int) -> str:
        if delay_minutes >= 60:
            return "speak with station staff if you have a connection, checked baggage, or a sleeper reservation affected by the delay."
        if delay_minutes > 0:
            return "remain at the station, watch the next stop update, and re-check after the next decision point."
        return "proceed normally and arrive at the platform before the posted boarding time."

    def _citations(self, query: str, top_k: int = 3) -> str:
        evidence = self.index.search(query, top_k=top_k)
        return ", ".join(doc["title"] for doc in evidence) or "Cancellation Recovery Playbook"

    def _cancellation_answer(self, flight_number: str, query: str) -> str:
        record = self.flights.get(flight_number)
        if not record:
            return (
                f"I could not find structured status for {flight_number}. "
                "I will not invent cancellation, refund, or alternate-flight details; please verify the flight number or carrier."
            )

        route_options = self.alternates.get(flight_number, {}).get("options", [])
        cheapest = min(route_options, key=lambda option: option["price_usd"]) if route_options else None
        option_lines = []
        for option in route_options:
            marker = " Cheapest sample option." if cheapest and option is cheapest else ""
            option_lines.append(
                f"{option['carrier']} {option['flight_number']} departs {option['departure']} and arrives {option['arrival']} "
                f"via {option['layover']}; {option['cabin']}; sample fare ${option['price_usd']}. {option['rank_note']}.{marker}"
            )
        options_text = " ".join(option_lines) if option_lines else "No protected alternate inventory is loaded yet for this route."

        citations = self._citations(query + " cancelled flight refund voucher compensation alternate flight")
        return (
            f"Recovery plan for {record['carrier']} flight {record['flight_number']} from {record['origin']} to {record['destination']}: "
            f"the flight is marked {record['status'].lower()} due to: {record['cause']}. "
            "First decision: choose rebooking if you still need to travel, or request a refund to the original payment method if you no longer want the trip. "
            "Ask the airline specifically about meal vouchers, hotel support if this becomes an overnight disruption, ground transport, lounge access, bonus miles, or travel credit, but treat those as policy or goodwill benefits rather than guaranteed compensation unless the carrier confirms them. "
            f"Alternate flights to evaluate: {options_text} "
            "Next 15 minutes: do not accept the first small voucher until you compare it with the refund path and any credit-card trip-delay benefit; save the cancellation notice, receipts, and screenshots of every offer; then confirm baggage transfer and connection time before accepting a reroute. "
            f"Sources: {citations}."
        )

    def _trip_plan_answer(self, query: str) -> str:
        citations = self._citations(query + " destination itinerary plan places food pacing", top_k=3)
        if "dubai" in query.lower():
            return (
                "Here is a balanced Dubai plan. Day 1: start in Old Dubai with Al Fahidi, Dubai Creek, an abra ride, and the souks, then keep the evening for Downtown and Burj Khalifa when the heat drops. "
                "Day 2: use the middle of the day for an indoor anchor such as Dubai Mall or a museum, then choose Jumeirah or Kite Beach near sunset. "
                "Day 3: pick one cluster only, Marina and JBR for a modern waterfront day or Palm Jumeirah for a slower premium day, and leave buffer for traffic. "
                "Food strategy: try Emirati or Levantine food near Old Dubai, Indian or Pakistani food in Deira or Karama for value, and one nicer dinner near Downtown or Marina. "
                f"TripSignal would next ask for your budget, hotel area, and whether you prefer heritage, shopping, beaches, or food so it can tighten the route. Sources: {citations}."
            )
        if "india" in query.lower() or "hyderabad" in query.lower():
            return (
                "For an India trip, I would pace it instead of overpacking it. Start with one arrival and jet-lag recovery block: daylight, hydration, a short walk, and an early local dinner. "
                "Then build each day around one anchor experience, one food highlight, and one flexible buffer for traffic or rest. "
                "If this is a first North India route, Delhi, Agra, and Jaipur work as the Golden Triangle. For South India, Hyderabad, Bengaluru, Chennai, Kerala, and Hampi need very different pacing, so the system should ask whether the priority is family, heritage, food, beaches, or business. "
                f"Before finalizing, TripSignal checks domestic flight versus train time, hotel neighborhood, monument closed days, and safe arrival time. Sources: {citations}."
            )
        citations = self._citations(query + " travel itinerary destination planning", top_k=3)
        return (
            "I can build that as a paced itinerary: one anchor experience per day, one food highlight, travel-time buffers, and a light first day if you are crossing time zones. "
            f"Tell me the destination, number of days, budget, and travel style, and I will turn it into a day-by-day plan. Sources: {citations}."
        )

    def _deals_answer(self, query: str) -> str:
        citations = self._citations(query + " flight deals fare strategy cheapest price baggage flexible", top_k=3)
        route_key = "IAD-HYD" if any(term in query.lower() for term in ["india", "hyderabad", "iad", "washington"]) else "DFW-ORD"
        deal_set = self.provider.get_deals(route_key) or {}
        options = deal_set.get("options", [])
        lines = []
        for option in options:
            lines.append(
                f"{option['carrier']} via {option['route']} departs {option['departure']} and arrives {option['arrival']}; sample fare ${option['price_usd']}; {option['notes']}."
            )
        cheapest = min(options, key=lambda item: item["price_usd"]) if options else None
        cheapest_text = f" Cheapest sample: {cheapest['carrier']} at ${cheapest['price_usd']}." if cheapest else ""
        return (
            f"For {deal_set.get('route_label', route_key)}, I would compare total value, not just the lowest sticker price. "
            + " ".join(lines)
            + cheapest_text
            + " Before choosing, check baggage, layover length, separate tickets, refund rules, loyalty earning, and whether a premium card adds trip protection or lounge value. "
            + f"These are sample fares until a live provider such as Amadeus or Duffel is connected. Sources: {citations}."
        )

    def _readiness_answer(self, query: str) -> str:
        citations = self._citations(query + " travel checklist passport visa baggage long haul readiness", top_k=3)
        return (
            "Here is the readiness check I would run before this trip: passport and transit visa status, onward boarding pass, baggage through-check, hotel address saved offline, eSIM or roaming plan, payment cards, medication in cabin baggage, chargers, downloaded entertainment, and a small delay kit in the carry-on. "
            "For a long-haul trip, add hydration, sleep timing, daylight on arrival, and a light first evening instead of dense sightseeing. "
            f"If you tell me your route and departure date, I can turn this into a timed checklist. Sources: {citations}."
        )

    def _layover_answer(self, query: str) -> str:
        citations = self._citations(query + " Dubai layover lounge credit card benefits sleep shower offers", top_k=3)
        return (
            "For an 8-hour Dubai layover on a U.S. to India itinerary, treat it as a planned recovery block. "
            "First 30 minutes: confirm the onward gate, baggage through-check, and whether your next boarding pass is already issued. "
            "Hours 1 to 3: use lounge access if you have it through the airline, alliance status, Priority Pass, or a premium travel card; eat a real meal, shower, hydrate, and charge devices. "
            "Hours 3 to 6: choose either sleep or a short experience. A sleep pod or airside hotel is the safer recommendation before another long-haul segment; a city trip only makes sense if your transit-visa status is clear and you can be back with a large buffer. "
            "Money checks: before buying a lounge pass, verify card lounge benefits, guest rules, shower access, and whether trip-delay coverage from the card used for booking could reimburse meals or a hotel. "
            "Final buffer: set two alarms if you sleep, keep passport and onward boarding pass together, and return to the gate area before boarding begins. "
            f"Sources: {citations}."
        )

    def answer(self, query: str) -> str:
        query_lower = query.lower()
        flight_match = FLIGHT_RE.search(query.upper())
        if flight_match:
            flight_number = flight_match.group(1)
            record = self.flights.get(flight_number)
            if "cancel" in query_lower or (record and record.get("status") == "Cancelled"):
                return self._cancellation_answer(flight_number, query)
            return self._flight_answer(flight_number, query) or ""

        if "dubai" in query_lower and any(term in query_lower for term in ["layover", "wait", "8 hour", "eight hour", "connection"]):
            return self._layover_answer(query)

        if any(term in query_lower for term in ["cheap flight", "cheapest", "deal", "deals", "fare", "flight price", "offers"]):
            return self._deals_answer(query)

        if any(term in query_lower for term in ["checklist", "packing", "prepare", "before my trip", "before my long-haul", "what should i check", "what should i carry"]):
            return self._readiness_answer(query)

        if any(term in query_lower for term in ["itinerary", "plan", "trip to", "visit", "what to do in", "days in", "travel plan"]):
            return self._trip_plan_answer(query)

        train_match = TRAIN_RE.search(query)
        asks_train = any(word in query.lower() for word in ["train", "amtrak", "rail", "station"])
        if train_match and asks_train:
            train_answer = self._train_answer(train_match.group(1), query)
            if train_answer:
                return train_answer

        evidence = self.index.search(query, top_k=3)
        if not evidence:
            return (
                "I do not have enough retrieved evidence to answer that safely. "
                "Please add a flight number, train number, route, or disruption type."
            )

        snippets = []
        citations = []
        for doc in evidence:
            body = " ".join(doc["text"].split())
            snippets.append(body[:420])
            citations.append(doc["title"])

        return (
            "Here is the grounded guidance I found: "
            + " ".join(snippets)
            + f" Sources: {', '.join(citations)}."
        )
