import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import TravelDisruptionAssistant


def main() -> None:
    assistant = TravelDisruptionAssistant()
    demos = [
        "Is AA184 delayed and what should I do?",
        "My EK232 flight was cancelled. What refund, voucher, compensation, and alternate flight options do I have?",
        "My UA920 flight is delayed. Will I miss my connection?",
        "I am traveling from America to India and have an 8 hour layover in Dubai. What can I do?",
        "What happens if Amtrak train 391 is late?",
        "What are my options if my flight is cancelled?",
    ]
    for query in demos:
        print("\nQ:", query)
        print("A:", assistant.answer(query))


if __name__ == "__main__":
    main()
