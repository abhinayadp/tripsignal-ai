import sys

from .chatbot import TravelDisruptionAssistant


def main() -> None:
    assistant = TravelDisruptionAssistant()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(assistant.answer(query))
        return

    print("TripSignal AI CLI. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        print("\nTripSignal:", assistant.answer(query))


if __name__ == "__main__":
    main()
