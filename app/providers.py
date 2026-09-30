import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STATUS_DIR = BASE_DIR / "data" / "status"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class SampleTravelDataProvider:
    """Sample provider with the same contract expected from live APIs.

    Production adapters can implement these methods with FlightAware AeroAPI,
    Cirium, AviationStack, Amadeus, Duffel, Amtrak GTFS, or agency GTFS-RT
    without changing the chatbot orchestration layer.
    """

    name = "sample"

    def __init__(self):
        self.flights = load_json(STATUS_DIR / "flights.json")
        self.trains = load_json(STATUS_DIR / "trains.json")
        self.alternates = load_json(STATUS_DIR / "alternate_flights.json")

    def get_flight(self, flight_number: str) -> dict | None:
        return self.flights.get(flight_number.upper())

    def get_train(self, train_key: str) -> dict | None:
        return self.trains.get(train_key) or self.trains.get(train_key.upper())

    def get_alternate_flights(self, flight_number: str) -> list[dict]:
        return self.alternates.get(flight_number.upper(), {}).get("options", [])


class LiveProviderNotConfigured(RuntimeError):
    pass
