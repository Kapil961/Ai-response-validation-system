import json
from collections import defaultdict
from pathlib import Path

from agents.judges.accuracy_judge import AccuracyJudge
from agents.judges.completeness_judge import CompletenessJudge
from agents.judges.hallucination_judge import HallucinationJudge
from agents.judges.relevance_judge import RelevanceJudge


class JudgeBenchmarkEvaluator:
    """Evaluates the four AI response validation judges."""

    def __init__(self):
        self.judges = {
            "accuracy": AccuracyJudge(embedding_model=None),
            "relevance": RelevanceJudge(embedding_model=None),
            "completeness": CompletenessJudge(embedding_model=None),
            "hallucination": HallucinationJudge(embedding_model=None),
        }

    @staticmethod
    def load_dataset(file_path):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        examples = dataset.get("examples")

        if not isinstance(examples, list) or not examples:
            raise ValueError("Benchmark dataset cannot be empty.")

        return examples

    def evaluate_example(self, example):
        example_id = example.get("id")
        judge_name = example.get("judge")

        if not example_id:
            raise ValueError("Benchmark example ID is missing.")

        if judge_name not in self.judges:
            raise ValueError(f"Unknown judge: {judge_name}")

        judge = self.judges[judge_name]
        response = example.get("response", "")

        if not isinstance(response, str) or not response.strip():
            raise ValueError("Response must be a non-empty string.")

        if judge_name == "accuracy":
            result = judge.evaluate(
                response=response,
                reference_answer=example.get("reference_answer"),
                source_document=example.get("source_document"),
            )

        elif judge_name == "relevance":
            result = judge.evaluate(
                question=example.get("question", ""),
                response=response,
            )

        elif judge_name == "completeness":
            result = judge.evaluate(
                question=example.get("question", ""),
                response=response,
            )

        else:
            result = judge.evaluate(
                response=response,
                source_document=example.get("source_document"),
            )

        actual_status = result.get("status", "unknown")
        expected_status = example.get("expected_status", "evaluated")

        record = {
            "id": example_id,
            "judge": judge_name,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "status_correct": actual_status == expected_status,
            "score": result.get("score"),
            "explanation_present": bool(
                result.get("explanation", "").strip()
            ),
        }

        if judge_name in ("accuracy", "relevance"):
            expected = example.get("expected_classification")
            actual = result.get("classification")

            record.update({
                "expected_classification": expected,
                "actual_classification": actual,
                "classification_correct": expected == actual,
            })

            record["evidence_present"] = bool(result.get("evidence"))

        elif judge_name == "completeness":
            score = result.get("score")
            minimum = example.get("expected_min_score")
            maximum = example.get("expected_max_score")

            score_correct = (
                score is not None
                and (minimum is None or score >= minimum)
                and (maximum is None or score <= maximum)
            )

            record.update({
                "expected_min_score": minimum,
                "expected_max_score": maximum,
                "score_band_correct": (
                    score_correct
                    if actual_status == "evaluated"
                    else None
                ),
                "covered_concepts": result.get(
                    "covered_concepts", []
                ),
                "missing_concepts": result.get(
                    "missing_concepts", []
                ),
            })

            # Evidence details are applicable to evaluated examples.
            # Insufficient-evidence cases are excluded from this metric.
            record["evidence_present"] = (
                (
                    "covered_concepts" in result
                    and "missing_concepts" in result
                )
                if actual_status == "evaluated"
                else None
            )

        else:
            claims = result.get("claims", [])
            expected_claims = example.get(
                "expected_claim_classifications", []
            )
            actual_claims = [
                claim.get("classification")
                for claim in claims
            ]

            claim_correct = (
                len(expected_claims) == len(actual_claims)
                and expected_claims == actual_claims
            )

            record.update({
                "expected_claim_classifications": expected_claims,
                "actual_claim_classifications": actual_claims,
                "claim_classifications_correct": (
                    claim_correct
                    if actual_status == "evaluated"
                    else None
                ),
                "claim_count": len(claims),
                "matched_evidence_available": (
                    bool(claims)
                    and all(
                        "matched_evidence_terms" in claim
                        for claim in claims
                    )
                ),
            })

            record["evidence_present"] = (
                record["matched_evidence_available"]
                if actual_status == "evaluated"
                else None
            )

        return record

    @staticmethod
    def classification_metrics(records):
        valid = [
            record
            for record in records
            if record.get("classification_correct") is not None
        ]

        if not valid:
            return {}

        labels = sorted({
            record["expected_classification"]
            for record in valid
        } | {
            record["actual_classification"]
            for record in valid
        })

        confusion = {
            label: {other: 0 for other in labels}
            for label in labels
        }

        for record in valid:
            expected = record["expected_classification"]
            actual = record["actual_classification"]
            confusion[expected][actual] += 1

        per_class = {}

        for label in labels:
            true_positive = confusion[label][label]
            false_positive = sum(
                confusion[other][label]
                for other in labels if other != label
            )
            false_negative = sum(
                confusion[label][other]
                for other in labels if other != label
            )

            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive else 0.0
            )

            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative else 0.0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            per_class[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        total = len(valid)
        accuracy = sum(
            record["classification_correct"]
            for record in valid
        ) / total

        return {
            "classification_accuracy": round(accuracy, 4),
            "macro_precision": round(
                sum(item["precision"] for item in per_class.values())
                / len(labels), 4
            ),
            "macro_recall": round(
                sum(item["recall"] for item in per_class.values())
                / len(labels), 4
            ),
            "macro_f1": round(
                sum(item["f1"] for item in per_class.values())
                / len(labels), 4
            ),
            "per_class": per_class,
            "confusion_matrix": confusion,
        }

    def evaluate_dataset(self, examples):
        if not examples:
            raise ValueError("Benchmark dataset cannot be empty.")

        results = [
            self.evaluate_example(example)
            for example in examples
        ]

        grouped = defaultdict(list)

        for result in results:
            grouped[result["judge"]].append(result)

        summary = {}

        for judge_name, records in grouped.items():
            total = len(records)

            evidence_records = [
                record for record in records
                if record["evidence_present"] is not None
            ]

            item = {
                "total_examples": total,
                "status_accuracy": round(
                    sum(r["status_correct"] for r in records) / total,
                    4,
                ),
                "explanation_coverage": round(
                    sum(r["explanation_present"] for r in records) / total,
                    4,
                ),
                "evidence_coverage": (
                    round(
                        sum(
                            r["evidence_present"]
                            for r in evidence_records
                        ) / len(evidence_records),
                        4,
                    )
                    if evidence_records
                    else None
                ),
                "evidence_examples_evaluated": len(evidence_records),
            }

            if judge_name in ("accuracy", "relevance"):
                item.update(self.classification_metrics(records))

            elif judge_name == "completeness":
                evaluated = [
                    r for r in records
                    if r["score_band_correct"] is not None
                ]

                item["score_band_accuracy"] = round(
                    sum(r["score_band_correct"] for r in evaluated)
                    / len(evaluated),
                    4,
                ) if evaluated else None

            else:
                evaluated = [
                    r for r in records
                    if r["claim_classifications_correct"] is not None
                ]

                item["claim_classification_accuracy"] = round(
                    sum(
                        r["claim_classifications_correct"]
                        for r in evaluated
                    ) / len(evaluated),
                    4,
                ) if evaluated else None

            summary[judge_name] = item

        return {
            "total_examples": len(results),
            "judge_summary": summary,
            "example_results": results,
        }


if __name__ == "__main__":
    dataset_path = (
        "datasets/processed/benchmarks/judge_benchmark.json"
    )

    evaluator = JudgeBenchmarkEvaluator()
    examples = evaluator.load_dataset(dataset_path)
    report = evaluator.evaluate_dataset(examples)

    print(json.dumps(report, indent=4))

    output_path = Path(
        "docs/reports/judge_benchmark_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nBenchmark report saved to: {output_path}")
