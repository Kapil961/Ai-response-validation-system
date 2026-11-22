
import pytest

from agents.judges.accuracy_judge import AccuracyJudge
from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def accuracy_judge():
    model = EmbeddingModel(device="cpu")
    return AccuracyJudge(embedding_model=model)


@pytest.mark.parametrize(
    "reference,response",
    [
        (
            "The capital of India is New Delhi.",
            "New Delhi serves as India's capital.",
        ),
        (
            "Machine learning enables computers to learn from data.",
            "Computers use machine learning to identify patterns in data.",
        ),
        (
            "Water freezes at zero degrees Celsius.",
            "At 0 degrees Celsius, water changes into ice.",
        ),
    ],
)
def test_real_model_paraphrased_answers(
    accuracy_judge, reference, response
):
    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["evidence"] is not None
    assert "Semantic similarity:" in result["evidence"]
    assert result["score"] >= 70
    assert result["passed"] is True

@pytest.mark.parametrize(
    "reference,response",
    [
        (
            "The capital of India is New Delhi.",
            "New Delhi is the capital of India and is the seat of the national government.",
        ),
        (
            "Python is a programming language.",
            "Python is a programming language widely used in data science and automation.",
        ),
    ],
)
def test_real_model_additional_details(
    accuracy_judge, reference, response
):
    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["score"] is not None
    assert result["score"] >= 85
    assert result["classification"] == "correct"

@pytest.mark.parametrize(
    "reference,response",
    [
        (
            "The capital of India is New Delhi.",
            "Mumbai is the capital of India, not New Delhi.",
        ),
        (
            "Water freezes at zero degrees Celsius.",
            "Water does not freeze at zero degrees Celsius.",
        ),
    ],
)
def test_real_model_negation_contradictions(
    accuracy_judge, reference, response
):
    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["classification"] == "incorrect"
    assert result["score"] <= 49.99
