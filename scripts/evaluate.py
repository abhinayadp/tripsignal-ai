import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import TravelDisruptionAssistant

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_PATH = BASE_DIR / "data" / "eval" / "questions.jsonl"

ARMS = ["bm25", "dense", "hybrid"]


def _recall_rr(retrieved_ids: list[str], gold_doc_ids: list[str], k: int = 3) -> tuple[float, float]:
    retrieved_ids = retrieved_ids[:k]
    hits = [doc_id for doc_id in retrieved_ids if doc_id in gold_doc_ids]
    recall = len(set(hits)) / len(set(gold_doc_ids))
    reciprocal_rank = 0.0
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_doc_ids:
            reciprocal_rank = 1.0 / idx
            break
    return recall, reciprocal_rank


def retrieval_metrics(assistant: TravelDisruptionAssistant, query: str, gold_doc_ids: list[str], k: int = 3) -> tuple[float, float]:
    """Hybrid-arm Recall@k / reciprocal rank. Empty gold is excluded by the caller."""
    if not gold_doc_ids:
        return 1.0, 1.0
    retrieved = assistant.index.search(query, top_k=k)
    return _recall_rr([doc["id"] for doc in retrieved], gold_doc_ids, k)


def arm_metrics(assistant: TravelDisruptionAssistant, query: str, gold_doc_ids: list[str], arm: str, k: int = 3) -> tuple[float, float]:
    retrieved = assistant.index.search_arm(query, arm, top_k=k)
    return _recall_rr([doc["id"] for doc in retrieved], gold_doc_ids, k)


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return "could not find" in lower and "will not invent" in lower


def main() -> None:
    assistant = TravelDisruptionAssistant()
    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    verified = 0
    gold_rows = 0
    arm_recalls: dict[str, list[float]] = {arm: [] for arm in ARMS}
    arm_mrrs: dict[str, list[float]] = {arm: [] for arm in ARMS}
    refusal_rows = 0
    refusal_correct = 0

    for row in rows:
        result = assistant.answer_with_trace(row["query"])
        answer = result["answer"]
        missing = [term for term in row["expected_terms"] if term.lower() not in answer.lower()]
        verified += int(result["trace"]["verification"]["supported"])
        ok = not missing
        passed += int(ok)

        recall = rr = None
        if row.get("gold_doc_ids"):
            gold_rows += 1
            for arm in ARMS:
                arm_recall, arm_rr = arm_metrics(assistant, row["query"], row["gold_doc_ids"], arm)
                arm_recalls[arm].append(arm_recall)
                arm_mrrs[arm].append(arm_rr)
            recall, rr = arm_recalls["hybrid"][-1], arm_mrrs["hybrid"][-1]

        if row["category"] == "refusal":
            refusal_rows += 1
            refusal_correct += int(is_refusal(answer))

        status = "PASS" if ok else "FAIL"
        metric_text = f"Recall@3={recall:.2f} RR={rr:.2f}" if recall is not None else "Recall@3=n/a (no gold docs)"
        print(f"{status} {row['category']}: {metric_text} verified={result['trace']['verification']['supported']}")
        if missing:
            print(f"  Missing terms: {missing}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print(f"\nAnswer coverage: {passed}/{len(rows)} = {passed / len(rows):.1%}")
    print(f"Mean Recall@3: {mean(arm_recalls['hybrid']):.3f} (hybrid arm, {gold_rows} gold-bearing rows; refusal/entity rows have no KB gold docs and are scored separately)")
    print(f"MRR@3: {mean(arm_mrrs['hybrid']):.3f}")
    print("\nPer-arm ablation (gold-bearing rows only):")
    print(f"  BM25-only : Recall@3={mean(arm_recalls['bm25']):.3f}  MRR@3={mean(arm_mrrs['bm25']):.3f}")
    print(f"  Dense-only: Recall@3={mean(arm_recalls['dense']):.3f}  MRR@3={mean(arm_mrrs['dense']):.3f}")
    print(f"  Hybrid    : Recall@3={mean(arm_recalls['hybrid']):.3f}  MRR@3={mean(arm_mrrs['hybrid']):.3f}")
    print(f"\nVerifier-supported answers: {verified}/{len(rows)} = {verified / len(rows):.1%}")
    if refusal_rows:
        print(f"Refusal accuracy: {refusal_correct}/{refusal_rows} = {refusal_correct / refusal_rows:.1%} (refusal-category rows answered with an explicit refusal)")


if __name__ == "__main__":
    main()
