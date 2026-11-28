import pytest

from evaluation.judge_benchmark import JudgeBenchmarkEvaluator


BENCHMARK_PATH = (
    "datasets/processed/benchmarks/judge_benchmark.json"
)


@pytest.fixture
def evaluator():
    return JudgeBenchmarkEvaluator()


def test_load_benchmark_dataset(evaluator):
    examples = evaluator.load_dataset(BENCHMARK_PATH)

    assert len(examples) == 12


def test_benchmark_covers_all_four_judges(evaluator):
    examples = evaluator.load_dataset(BENCHMARK_PATH)

    judges = {example["judge"] for example in examples}

    assert judges == {
        "accuracy",
        "relevance",
        "completeness",
        "hallucination",
    }


def test_all_benchmark_statuses_match(evaluator):
    examples = evaluator.load_dataset(BENCHMARK_PATH)
    report = evaluator.evaluate_dataset(examples)

    for result in report["example_results"]:
        assert result["status_correct"] is True


def test_classification_metrics_are_correct(evaluator):
    records = [
        {
            "expected_classification": "correct",
            "actual_classification": "correct",
            "classification_correct": True,
        },
        {
            "expected_classification": "incorrect",
            "actual_classification": "incorrect",
            "classification_correct": True,
        },
    ]

    metrics = evaluator.classification_metrics(records)

    assert metrics["classification_accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_confusion_matrix_counts_misclassification(evaluator):
    records = [
        {
            "expected_classification": "correct",
            "actual_classification": "incorrect",
            "classification_correct": False,
        },
        {
            "expected_classification": "incorrect",
            "actual_classification": "incorrect",
            "classification_correct": True,
        },
    ]

    metrics = evaluator.classification_metrics(records)

    assert metrics["confusion_matrix"]["correct"]["incorrect"] == 1
    assert metrics["confusion_matrix"]["incorrect"]["incorrect"] == 1
    assert metrics["classification_accuracy"] == 0.5


def test_rejects_empty_benchmark_dataset(evaluator):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate_dataset([])


def test_rejects_unknown_judge(evaluator):
    example = {
        "id": "INVALID-001",
        "judge": "unknown",
        "response": "Test response.",
    }

    with pytest.raises(ValueError, match="Unknown judge"):
        evaluator.evaluate_example(example)


def test_rejects_missing_response(evaluator):
    example = {
        "id": "INVALID-002",
        "judge": "accuracy",
        "response": "   ",
    }

    with pytest.raises(ValueError, match="non-empty string"):
        evaluator.evaluate_example(example)
