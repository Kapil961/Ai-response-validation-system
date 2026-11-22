
import pytest

from agents.judges.accuracy_judge import AccuracyJudge
from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def accuracy_judge():
    model = EmbeddingModel(device="cpu")
    return AccuracyJudge(embedding_model=model)


def test_real_model_paraphrased_answer(accuracy_judge):
    reference = "The capital of India is New Delhi."
    response = "New Delhi serves as India's capital."

    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["evidence"] is not None
    assert "Semantic similarity:" in result["evidence"]


def test_real_model_additional_correct_details(accuracy_judge):
    reference = "The capital of India is New Delhi."
    response = (
        "New Delhi is the capital of India "
        "and is the seat of the national government."
    )

    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["score"] is not None
    assert "Semantic similarity:" in result["evidence"]


def test_real_model_contradictory_answer(accuracy_judge):
    reference = "The capital of India is New Delhi."
    response = "Mumbai is the capital of India, not New Delhi."

    result = accuracy_judge.evaluate(response, reference)

    assert result["status"] == "evaluated"
    assert result["classification"] == "incorrect"
    assert result["score"] <= 49.99
