import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import TravelDisruptionAssistant

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_PATH = BASE_DIR / "data" / "eval" / "questions.jsonl"


def main() -> None:
    assistant = TravelDisruptionAssistant()
    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    for row in rows:
        answer = assistant.answer(row["query"])
        missing = [term for term in row["expected_terms"] if term.lower() not in answer.lower()]
        ok = not missing
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['category']}: {row['query']}")
        if missing:
            print(f"  Missing terms: {missing}")
    print(f"\nAnswer coverage: {passed}/{len(rows)} = {passed / len(rows):.1%}")


if __name__ == "__main__":
    main()
