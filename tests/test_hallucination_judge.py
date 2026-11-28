
from agents.judges.hallucination_judge import HallucinationJudge


def test_supported_claim():
    judge = HallucinationJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data.",
        source_document=(
            "Machine learning is a branch of artificial intelligence "
            "that enables computers to learn from data."
        )
    )

    assert result["status"] == "evaluated"
    assert result["claims"][0]["classification"] == "supported"


def test_unsupported_claim():
    judge = HallucinationJudge()

    result = judge.evaluate(
        response="Machine learning was invented in 1905.",
        source_document=(
            "Machine learning enables computers to learn from data."
        )
    )

    assert result["status"] == "evaluated"
    assert result["claims"][0]["classification"] == "potentially_unsupported"


def test_missing_source():
    judge = HallucinationJudge()

    result = judge.evaluate(
        response="Machine learning enables computers to learn from data."
    )

    assert result["status"] == "insufficient_evidence"
    assert result["score"] is None


def test_multiple_claims():
    judge = HallucinationJudge()

    result = judge.evaluate(
        response=(
            "Machine learning enables computers to learn from data. "
            "The moon is made of cheese."
        ),
        source_document=(
            "Machine learning enables computers to learn from data."
        )
    )

    assert len(result["claims"]) == 2
    assert result["claims"][0]["classification"] == "supported"
    assert result["claims"][1]["classification"] == "potentially_unsupported"
def test_contradictory_claim():
    judge = HallucinationJudge()

    result = judge.evaluate(
        response=(
            "Machine learning does not enable "
            "computers to learn from data."
        ),
        source_document=(
            "Machine learning enables computers "
            "to learn from data."
        )
    )

    assert result["status"] == "evaluated"
    assert result["claims"][0]["classification"] == "contradictory"

def test_numbered_list_items_are_split_correctly():
    judge = HallucinationJudge()

    response = (
        "1. Leading the Council of Ministers.\n"
        "2. Coordinating the work of different ministries.\n"
        "3. Representing India in international meetings."
    )

    sentences = judge.split_sentences(response)

    assert len(sentences) == 3
    assert sentences[0] == "Leading the Council of Ministers."
    assert sentences[1] == "Coordinating the work of different ministries."
    assert sentences[2] == "Representing India in international meetings."
