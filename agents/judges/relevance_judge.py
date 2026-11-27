
import re

from typing import Optional


class RelevanceJudge:

    def __init__(
        self,
        passing_threshold: float = 50.0,
        embedding_model=None,
        lexical_weight: float = 0.60,
        semantic_weight: float = 0.40,
    ):
        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError("Scoring weights cannot be negative.")

        if lexical_weight + semantic_weight <= 0:
            raise ValueError("At least one scoring weight must be positive.")

        self.passing_threshold = passing_threshold
        self.embedding_model = embedding_model
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

        self.stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "in", "on", "to", "for", "and",
            "how", "why", "when", "where", "does",
            "do", "can", "could", "would", "should",
            "explain", "describe", "tell", "me", "about",
            "please", "give", "provide"
        }

    def tokenize(self, text: str) -> set:

        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def calculate_semantic_similarity(
        self,
        question: str,
        response: str,
    ) -> Optional[float]:

        if self.embedding_model is None:
            return None

        if not question.strip() or not response.strip():
            return None

        try:
            embeddings = self.embedding_model.encode(
                [question, response]
            )

            similarity = sum(
                float(a) * float(b)
                for a, b in zip(embeddings[0], embeddings[1])
            )

            similarity = max(-1.0, min(1.0, similarity))

            return max(0.0, similarity) * 100

        except Exception:
            return None

    def evaluate(self, question: str, response: str) -> dict:

        question_words = self.tokenize(question)
        response_words = self.tokenize(response)

        if not question_words:
            return {
                "judge_name": "Relevance Judge",
                "score": None,
                "passed": None,
                "classification": "insufficient_evidence",
                "explanation": (
                    "No meaningful concepts could be extracted "
                    "from the question."
                ),
                "matched_concepts": [],
                "missing_concepts": [],
                "evidence": None,
                "status": "insufficient_evidence"
            }

        matched = question_words.intersection(response_words)
        missing = question_words - matched

        lexical_score = (
            len(matched) / len(question_words) * 100
        )

        semantic_score = self.calculate_semantic_similarity(
            question,
            response,
        )

        if semantic_score is not None:

            total_weight = (
                self.lexical_weight + self.semantic_weight
            )

            score = (
                self.lexical_weight * lexical_score
                + self.semantic_weight * semantic_score
            ) / total_weight

            scoring_method = "lexical and semantic matching"

        else:

            score = lexical_score
            scoring_method = "lexical matching only"

        score = round(score, 2)

        if score >= 80:
            classification = "fully_relevant"
        elif score >= 40:
            classification = "partially_relevant"
        elif score > 0:
            classification = "weakly_relevant"
        else:
            classification = "unrelated"

        explanation = (
            f"The response was evaluated using {scoring_method}. "
            f"It matched {len(matched)} out of "
            f"{len(question_words)} meaningful question concepts. "
            f"The final relevance score is {score}%. "
        )

        if semantic_score is not None:
            explanation += (
                f"Lexical score: {lexical_score:.2f}%. "
                f"Semantic score: {semantic_score:.2f}%. "
            )

        if matched:
            explanation += (
                "Matched concepts: "
                + ", ".join(sorted(matched)) + ". "
            )

        if missing:
            explanation += (
                "Missing concepts: "
                + ", ".join(sorted(missing)) + "."
            )

        return {
            "judge_name": "Relevance Judge",
            "score": score,
            "passed": score >= self.passing_threshold,
            "classification": classification,
            "explanation": explanation,
            "matched_concepts": sorted(matched),
            "missing_concepts": sorted(missing),
            "evidence": (
                f"Matched {len(matched)} out of "
                f"{len(question_words)} question concepts."
            ),
            "status": "evaluated"
        }
