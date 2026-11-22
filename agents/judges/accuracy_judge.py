
import re
from collections import Counter
from typing import Optional


class AccuracyJudge:

    NEGATION_TERMS = {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "can't",
        "without",
        "neither",
        "nor",
    }

    def __init__(
        self,
        passing_threshold: float = 70.0,
        embedding_model=None,
    ):
        self.passing_threshold = passing_threshold
        self.embedding_model = embedding_model

    def tokenize(self, text: str) -> list:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def calculate_semantic_similarity(
        self,
        response: str,
        reference_answer: str,
    ) -> Optional[float]:
        """
        Calculate cosine similarity using the existing embedding model.
        Returns a score between 0 and 100.
        """

        if self.embedding_model is None:
            return None

        try:
            embeddings = self.embedding_model.encode(
                [response, reference_answer]
            )

            similarity = float(
                sum(
                    float(a) * float(b)
                    for a, b in zip(embeddings[0], embeddings[1])
                )
            )

            similarity = max(-1.0, min(1.0, similarity))

            return round(max(0.0, similarity) * 100, 2)

        except Exception:
            # Preserve lexical evaluation if embedding calculation fails.
            return None

    def evaluate(
        self,
        response: str,
        reference_answer: Optional[str] = None,
    ) -> dict:

        if not reference_answer or not reference_answer.strip():
            return {
                "judge_name": "Accuracy Judge",
                "score": None,
                "passed": None,
                "classification": "insufficient_evidence",
                "explanation": (
                    "Accuracy cannot be estimated because no "
                    "reference answer was provided."
                ),
                "evidence": None,
                "matched_concepts": [],
                "missing_concepts": [],
                "status": "insufficient_evidence",
            }

        response_tokens = self.tokenize(response)
        reference_tokens = self.tokenize(reference_answer)

        if not response_tokens or not reference_tokens:
            return {
                "judge_name": "Accuracy Judge",
                "score": 0.0,
                "passed": False,
                "classification": "incorrect",
                "explanation": (
                    "Response or reference answer contains "
                    "no meaningful tokens."
                ),
                "evidence": None,
                "matched_concepts": [],
                "missing_concepts": [],
                "status": "evaluated",
            }

        # Calculate lexical F1.
        response_counts = Counter(response_tokens)
        reference_counts = Counter(reference_tokens)

        common = sum(
            (response_counts & reference_counts).values()
        )

        precision = common / len(response_tokens)
        recall = common / len(reference_tokens)

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2 * precision * recall
            ) / (precision + recall)

        lexical_score = f1_score * 100

        # Calculate semantic similarity using the shared model.
        semantic_score = self.calculate_semantic_similarity(
            response,
            reference_answer,
        )

        # Combine lexical and semantic scores when embeddings are available.
        if semantic_score is not None:
            score = (
                0.80 * lexical_score
                + 0.20 * semantic_score
            )
            scoring_method = "combined lexical F1 and semantic similarity"
        else:
            score = lexical_score
            scoring_method = "lexical F1 only"

        # Basic negation-mismatch safeguard.
        response_has_negation = bool(
            set(response_tokens) & self.NEGATION_TERMS
        )
        reference_has_negation = bool(
            set(reference_tokens) & self.NEGATION_TERMS
        )

        negation_mismatch = (
            response_has_negation != reference_has_negation
        )

        if negation_mismatch:
            score = min(score, 49.99)

        score = round(score, 2)

        # Classify the result.
        if score >= 85:
            classification = "correct"
        elif score >= 50:
            classification = "partially_correct"
        else:
            classification = "incorrect"

        matched = sorted(
            set(response_tokens).intersection(reference_tokens)
        )

        missing = sorted(
            set(reference_tokens) - set(response_tokens)
        )

        explanation = (
            f"The response achieved {score}% using "
            f"{scoring_method}. "
            f"Classification: {classification}. "
        )

        if semantic_score is not None:
            explanation += (
                f"Lexical F1: {round(lexical_score, 2)}%; "
                f"semantic similarity: {semantic_score}%. "
            )

        if negation_mismatch:
            explanation += (
                "A possible negation mismatch was detected. "
                "The score was capped conservatively. "
            )

        explanation += (
            "These automated comparisons do not independently "
            "establish factual correctness or detect all contradictions."
        )

        return {
            "judge_name": "Accuracy Judge",
            "score": score,
            "passed": score >= self.passing_threshold,
            "classification": classification,
            "explanation": explanation,
            "evidence": (
                f"Matched tokens: {common}; "
                f"Response tokens: {len(response_tokens)}; "
                f"Reference tokens: {len(reference_tokens)}; "
                f"Lexical F1: {round(lexical_score, 2)}%; "
                f"Semantic similarity: "
                f"{semantic_score if semantic_score is not None else 'unavailable'}"
            ),
            "matched_concepts": matched,
            "missing_concepts": missing,
            "status": "evaluated",
        }
