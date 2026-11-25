
import pytest

from agents.judges.hallucination_judge import HallucinationJudge
from rag.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def hallucination_judge():
    model = EmbeddingModel(device="cpu")
    return HallucinationJudge(embedding_model=model)


@pytest.mark.parametrize(
    "source, response",
    [
        (
            "Machine learning enables computers to learn from data.",
            "Computers use machine learning to identify patterns in data.",
        ),
        (
            "Water freezes at zero degrees Celsius.",
            "At 0 degrees Celsius, water changes into ice.",
        ),
        (
            "The capital of India is New Delhi.",
            "New Delhi serves as India's capital.",
        ),
    ],
)
def test_real_model_paraphrased_claims(
    hallucination_judge, source, response
):
    result = hallucination_judge.evaluate(
        response=response,
        source_document=source,
    )

    assert result["status"] == "evaluated"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["classification"] == "supported"
    assert result["claims"][0]["support_score"] >= 70


@pytest.mark.parametrize(
    "source, response",
    [
        (
            "Machine learning enables computers to learn from data.",
            "Machine learning enables computers to learn from data.",
        ),
        (
            "Python is a programming language.",
            "Python is a programming language used for automation.",
        ),
    ],
)
def test_real_model_supported_claims(
    hallucination_judge, source, response
):
    result = hallucination_judge.evaluate(
        response=response,
        source_document=source,
    )

    assert result["status"] == "evaluated"
    assert result["claims"][0]["classification"] == "supported"


@pytest.mark.parametrize(
    "source, response",
    [
        (
            "Machine learning enables computers to learn from data.",
            "The moon is made entirely of cheese.",
        ),
        (
            "Water freezes at zero degrees Celsius.",
            "Jupiter has 95 confirmed moons.",
        ),
    ],
)
def test_real_model_unsupported_claims(
    hallucination_judge, source, response
):
    result = hallucination_judge.evaluate(
        response=response,
        source_document=source,
    )

    assert result["status"] == "evaluated"
    assert (
        result["claims"][0]["classification"]
        == "potentially_unsupported"
    )


def test_real_model_negation_contradiction(
    hallucination_judge,
):
    result = hallucination_judge.evaluate(
        response="Water does not freeze at zero degrees Celsius.",
        source_document="Water freezes at zero degrees Celsius.",
    )

    assert result["status"] == "evaluated"
    assert result["claims"][0]["classification"] == "contradictory"
