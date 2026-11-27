
from agents.judges.completeness_judge import CompletenessJudge


def test_complete_response():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning is a field of AI. "
            "Its types include supervised and unsupervised learning. "
            "Applications include healthcare and finance."
        )
    )

    assert result["score"] == 100
    assert result["passed"] is True
    assert result["status"] == "evaluated"


def test_incomplete_response():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="Explain machine learning, its types, and applications.",
        response=(
            "Machine learning enables computers "
            "to learn patterns from data."
        )
    )

    assert result["score"] < 70
    assert result["passed"] is False
    assert "types" in result["missing_concepts"]
    assert "applications" in result["missing_concepts"]


def test_question_without_meaningful_concepts():
    judge = CompletenessJudge()

    result = judge.evaluate(
        question="What is it?",
        response="It is a system."
    )

    assert result["score"] is None
    assert result["passed"] is None
    assert result["status"] == "insufficient_evidence"

class FakeEmbeddingModel:

    def encode(self, texts):
        embeddings = {
            "automobiles": [1.0, 0.0],
            "cars provide convenient personal transportation.": [0.9, 0.1],
            "advantages": [0.0, 1.0],
            "cars": [0.9, 0.1],
        }

        return [
            embeddings.get(text.lower(), [0.0, 1.0])
            for text in texts
        ]


def test_semantic_matching_recognizes_paraphrase():

    judge = CompletenessJudge(
        embedding_model=FakeEmbeddingModel()
    )

    result = judge.evaluate(
        question="Explain automobiles.",
        response="Cars provide convenient personal transportation."
    )

    assert "automobiles" in result["covered_concepts"]
    assert result["score"] == 100.0


def test_semantic_matching_identifies_missing_concept():

    judge = CompletenessJudge(
        embedding_model=FakeEmbeddingModel()
    )

    result = judge.evaluate(
        question="Explain automobiles and advantages.",
        response="Cars provide convenient personal transportation."
    )

    assert "automobiles" in result["covered_concepts"]
    assert "advantages" in result["missing_concepts"]
    assert result["score"] == 50.0


def test_lexical_fallback_without_embedding_model():

    judge = CompletenessJudge()

    result = judge.evaluate(
        question="Explain machine learning.",
        response="Machine learning enables computers to learn."
    )

    assert result["score"] == 100.0
    assert result["passed"] is True
    assert "lexical matching only" in result["explanation"]
