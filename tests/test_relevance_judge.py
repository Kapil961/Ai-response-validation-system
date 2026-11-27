
import pytest

from agents.judges.relevance_judge import RelevanceJudge


def test_fully_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning is a branch of artificial intelligence."
    )

    assert result["classification"] == "fully_relevant"
    assert result["score"] == 100.0
    assert result["passed"] is True


def test_partially_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning uses data."
    )

    assert result["classification"] == "partially_relevant"
    assert result["score"] == 50.0


def test_weakly_relevant_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine."
    )

    assert result["classification"] == "weakly_relevant"
    assert result["score"] == 25.0


def test_unrelated_response():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning?",
        "The capital of France is Paris."
    )

    assert result["classification"] == "unrelated"
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_relevance_evidence_fields():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is machine learning and artificial intelligence?",
        "Machine learning uses data."
    )

    assert "machine" in result["matched_concepts"]
    assert "learning" in result["matched_concepts"]
    assert "artificial" in result["missing_concepts"]
    assert result["explanation"]
    assert result["evidence"]


def test_empty_question_returns_insufficient_evidence():
    judge = RelevanceJudge()

    result = judge.evaluate(
        "What is?",
        "Machine learning uses data."
    )

    assert result["classification"] == "insufficient_evidence"
    assert result["score"] is None
    assert result["status"] == "insufficient_evidence"


class FakeEmbeddingModel:

    def encode(self, texts):

        embeddings = {
            "what are the benefits of automobiles?": [1.0, 0.0],
            "cars provide convenient personal transportation.": [0.9, 0.1],
            "what is machine learning?": [1.0, 0.0],
            "the capital of france is paris.": [0.0, 1.0],
        }

        return [
            embeddings.get(text.lower(), [0.0, 1.0])
            for text in texts
        ]


def test_semantic_matching_recognizes_paraphrase():

    judge = RelevanceJudge(
        embedding_model=FakeEmbeddingModel()
    )

    result = judge.evaluate(
        question="What are the benefits of automobiles?",
        response="Cars provide convenient personal transportation."
    )

    assert result["score"] > 0.0
    assert result["classification"] == "weakly_relevant"


def test_semantic_matching_rejects_unrelated_response():

    judge = RelevanceJudge(
        embedding_model=FakeEmbeddingModel()
    )

    result = judge.evaluate(
        question="What is machine learning?",
        response="The capital of France is Paris."
    )

    assert result["score"] == 0.0
    assert result["classification"] == "unrelated"


def test_lexical_fallback_without_embedding_model():

    judge = RelevanceJudge()

    result = judge.evaluate(
        question="What is machine learning?",
        response="Machine learning uses data."
    )

    assert result["score"] == 100.0
    assert result["passed"] is True
    assert "lexical matching only" in result["explanation"]


def test_invalid_scoring_weights():

    with pytest.raises(ValueError):
        RelevanceJudge(
            lexical_weight=-0.1,
            semantic_weight=0.4,
        )


def test_zero_scoring_weights():

    with pytest.raises(ValueError):
        RelevanceJudge(
            lexical_weight=0.0,
            semantic_weight=0.0,
        )
