
import re

from collections import Counter
from typing import Optional


class AccuracyJudge:

    NEGATION_TERMS = {
        "not", "no", "never", "none", "cannot",
        "can't", "without", "neither", "nor",
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
            return None

    def insufficient_evidence(self) -> dict:
        return {
            "judge_name": "Accuracy Judge",
            "score": None,
            "passed": None,
            "classification": "insufficient_evidence",
            "explanation": (
                "Accuracy cannot be estimated because neither "
                "a reference answer nor source evidence was provided."
            ),
            "evidence": None,
            "matched_concepts": [],
            "missing_concepts": [],
            "status": "insufficient_evidence",
        }

    def evaluate_against_reference(
        self,
        response: str,
        reference_answer: str,
    ) -> dict:

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

        response_counts = Counter(response_tokens)
        reference_counts = Counter(reference_tokens)

        common = sum(
            (response_counts & reference_counts).values()
        )

        lexical_recall = (
            common / len(reference_tokens)
        ) * 100

        semantic_score = self.calculate_semantic_similarity(
            response,
            reference_answer,
        )

        if semantic_score is not None:
            score = (
                0.60 * lexical_recall
                + 0.40 * semantic_score
            )
            scoring_method = (
                "combined lexical recall and semantic similarity"
            )
        else:
            score = lexical_recall
            scoring_method = "lexical recall only"

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
                f"Lexical recall: {round(lexical_recall, 2)}%; "
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
                f"Lexical recall: {round(lexical_recall, 2)}%; "
                f"Semantic similarity: "
                f"{semantic_score if semantic_score is not None else 'unavailable'}"
            ),
            "matched_concepts": matched,
            "missing_concepts": missing,
            "status": "evaluated",
        }

    def evaluate_against_source(
        self,
        response: str,
        source_document: str,
    ) -> dict:

        response_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", response)
            if sentence.strip()
        ]

        source_sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                source_document,
            )
            if sentence.strip()
        ]

        if not response_sentences or not source_sentences:
            return self.insufficient_evidence()

        sentence_scores = []
        all_matched = set()
        all_missing = set()
        evidence_details = []

        for response_sentence in response_sentences:

            response_tokens = self.tokenize(response_sentence)

            if not response_tokens:
                continue

            best_score = -1.0
            best_source_tokens = []

            for source_sentence in source_sentences:

                source_tokens = self.tokenize(source_sentence)

                if not source_tokens:
                    continue

                common = set(response_tokens).intersection(source_tokens)

                lexical_score = (
                    len(common) / len(set(source_tokens))
                ) * 100

                semantic_score = self.calculate_semantic_similarity(
                    response_sentence,
                    source_sentence,
                )

                if semantic_score is not None:
                    candidate_score = (
                        0.60 * lexical_score
                        + 0.40 * semantic_score
                    )
                else:
                    candidate_score = lexical_score

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_source_tokens = source_tokens

            if best_score < 0:
                continue

            response_negation = bool(
                set(response_tokens) & self.NEGATION_TERMS
            )

            source_negation = bool(
                set(best_source_tokens) & self.NEGATION_TERMS
            )

            if response_negation != source_negation:
                best_score = min(best_score, 49.99)

            best_score = round(best_score, 2)

            sentence_scores.append(best_score)

            matched = (
                set(response_tokens) & set(best_source_tokens)
            )

            missing = (
                set(best_source_tokens) - set(response_tokens)
            )

            all_matched.update(matched)
            all_missing.update(missing)

            evidence_details.append(
                f"Response sentence: {response_sentence} "
                f"| Best source match score: {best_score}%"
            )

        if not sentence_scores:
            return self.insufficient_evidence()

        score = round(
            sum(sentence_scores) / len(sentence_scores),
            2,
        )

        # Detect unsupported claims independently of the overall average.
        has_unsupported_claim = any(
            sentence_score < 50
            for sentence_score in sentence_scores
        )

        if score >= 85 and not has_unsupported_claim:
            classification = "correct"
        elif score >= 50:
            classification = "partially_correct"
        else:
            classification = "incorrect"

        explanation = (
            f"The response achieved {score}% based on "
            f"sentence-level comparisons with source evidence. "
            f"Classification: {classification}. "
        )

        if has_unsupported_claim:
            explanation += (
                "At least one response sentence had a source-match "
                "score below 50%. The response is therefore not "
                "considered fully supported. "
            )

        explanation += (
            "This is an automated similarity-based assessment, "
            "not independent factual verification."
        )

        return {
            "judge_name": "Accuracy Judge",
            "score": score,
            "passed": (
                score >= self.passing_threshold
                and not has_unsupported_claim
            ),
            "classification": classification,
            "explanation": explanation,
            "evidence": (
                "Source-based evaluation; "
                + "; ".join(evidence_details)
            ),
            "matched_concepts": sorted(all_matched),
            "missing_concepts": sorted(all_missing),
            "status": "evaluated",
        }

    def evaluate(
        self,
        response: str,
        reference_answer: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> dict:

        if reference_answer and reference_answer.strip():
            return self.evaluate_against_reference(
                response,
                reference_answer,
            )

        if source_document and source_document.strip():
            return self.evaluate_against_source(
                response,
                source_document,
            )

        return self.insufficient_evidence()
