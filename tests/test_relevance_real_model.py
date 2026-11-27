
import pytest

from agents.judges.relevance_judge import RelevanceJudge
from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def relevance_judge():
    model = EmbeddingModel(device="cpu")
    return RelevanceJudge(embedding_model=model)


def test_real_model_fully_relevant_response(relevance_judge):

    result = relevance_judge.evaluate(
        question="What is machine learning?",
        response=(
            "Machine learning is a branch of artificial intelligence "
            "that enables computers to learn from data."
        ),
    )

    assert result["status"] == "evaluated"
    assert result["score"] >= 80.0
    assert result["classification"] == "fully_relevant"
    assert result["passed"] is True


def test_real_model_paraphrased_response(relevance_judge):

    result = relevance_judge.evaluate(
        question="What are the benefits of automobiles?",
        response=(
            "Cars provide convenient personal transportation."
        ),
    )

    assert result["status"] == "evaluated"
    assert result["score"] > 0.0
    assert result["score"] < 40.0
    assert result["classification"] == "weakly_relevant"


def test_real_model_unrelated_response(relevance_judge):

    result = relevance_judge.evaluate(
        question="What is machine learning?",
        response="The capital of France is Paris.",
    )

    assert result["status"] == "evaluated"
    assert result["score"] < 40.0
    assert result["passed"] is False
