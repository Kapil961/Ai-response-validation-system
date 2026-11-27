
import pytest

from agents.judges.completeness_judge import CompletenessJudge
from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def completeness_judge():
    model = EmbeddingModel(device="cpu")
    return CompletenessJudge(embedding_model=model)


def test_real_model_complete_response(completeness_judge):

    result = completeness_judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning is a field of AI. "
            "Its types include supervised and unsupervised learning. "
            "Applications include healthcare and finance."
        ),
    )

    assert result["status"] == "evaluated"
    assert result["score"] == 100.0
    assert result["passed"] is True
    assert result["missing_concepts"] == []


def test_real_model_incomplete_response(completeness_judge):

    result = completeness_judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning enables computers "
            "to learn patterns from data."
        ),
    )

    assert result["status"] == "evaluated"
    assert result["score"] == 50.0
    assert result["passed"] is False
    assert "types" in result["missing_concepts"]
    assert "applications" in result["missing_concepts"]


def test_real_model_paraphrased_concept(completeness_judge):

    result = completeness_judge.evaluate(
        question="Explain the advantages of automobiles.",
        response="Cars provide convenient personal transportation.",
    )

    assert result["status"] == "evaluated"
    assert "automobiles" in result["covered_concepts"]
    assert "advantages" in result["missing_concepts"]
    assert result["score"] == 50.0
