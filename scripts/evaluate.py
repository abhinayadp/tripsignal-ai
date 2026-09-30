import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import TravelDisruptionAssistant

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_PATH = BASE_DIR / "data" / "eval" / "questions.jsonl"


def retrieval_metrics(assistant: TravelDisruptionAssistant, query: str, gold_doc_ids: list[str], k: int = 3) -> tuple[float, float]:
    if not gold_doc_ids:
        return 1.0, 1.0
    retrieved = assistant.index.search(query, top_k=k)
    retrieved_ids = [doc["id"] for doc in retrieved]
    hits = [doc_id for doc_id in retrieved_ids if doc_id in gold_doc_ids]
    recall = len(set(hits)) / len(set(gold_doc_ids))
    reciprocal_rank = 0.0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_doc_ids:
            reciprocal_rank = 1.0 / idx
            break
    return recall, reciprocal_rank


def main() -> None:
    assistant = TravelDisruptionAssistant()
    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    recalls = []
    mrrs = []
    verified = 0
    for row in rows:
        result = assistant.answer_with_trace(row["query"])
        answer = result["answer"]
        missing = [term for term in row["expected_terms"] if term.lower() not in answer.lower()]
        recall, rr = retrieval_metrics(assistant, row["query"], row.get("gold_doc_ids", []))
        recalls.append(recall)
        mrrs.append(rr)
        verified += int(result["trace"]["verification"]["supported"])
        ok = not missing
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['category']}: Recall@3={recall:.2f} RR={rr:.2f} verified={result['trace']['verification']['supported']}")
        if missing:
            print(f"  Missing terms: {missing}")
    print(f"\nAnswer coverage: {passed}/{len(rows)} = {passed / len(rows):.1%}")
    print(f"Mean Recall@3: {sum(recalls) / len(recalls):.3f}")
    print(f"MRR@3: {sum(mrrs) / len(mrrs):.3f}")
    print(f"Verifier-supported answers: {verified}/{len(rows)} = {verified / len(rows):.1%}")


if __name__ == "__main__":
    main()
