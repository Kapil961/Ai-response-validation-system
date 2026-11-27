
import re

from typing import Optional


class CompletenessJudge:

    def __init__(
        self,
        completeness_threshold: float = 70.0,
        embedding_model=None,
        semantic_threshold: float = 0.60,
    ):
        if not 0 <= semantic_threshold <= 1:
            raise ValueError(
                "semantic_threshold must be between 0 and 1."
            )

        self.completeness_threshold = completeness_threshold
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold

        self.stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "in", "on", "to", "for", "and",
            "how", "why", "when", "where", "does",
            "do", "can", "could", "would", "should",
            "explain", "describe", "discuss", "define",
            "tell", "me", "about", "give", "provide",
            "it", "its", "this", "that", "these", "those",
            "he", "she", "they", "we", "you", "i",
            "be", "been", "being", "was", "were",
            "has", "have", "had", "your", "my"
        }

    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            word for word in words
            if word not in self.stop_words
        }

    def calculate_semantic_similarity(
        self,
        concept: str,
        response_sentences: list,
    ) -> Optional[float]:

        if self.embedding_model is None or not response_sentences:
            return None

        try:
            embeddings = self.embedding_model.encode(
                [concept] + response_sentences
            )

            similarities = [
                sum(
                    float(a) * float(b)
                    for a, b in zip(embeddings[0], sentence_embedding)
                )
                for sentence_embedding in embeddings[1:]
            ]

            return max(
                max(-1.0, min(1.0, score))
                for score in similarities
            )

        except Exception:
            return None

    def evaluate(self, question: str, response: str) -> dict:

        expected_concepts = self.tokenize(question)
        response_concepts = self.tokenize(response)

        if not expected_concepts:
            return {
                "judge_name": "Completeness Judge",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "No meaningful concepts could be extracted "
                    "from the question."
                ),
                "missing_concepts": [],
                "covered_concepts": []
            }

        response_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", response)
            if sentence.strip()
        ]

        covered_concepts = set()
        missing_concepts = set()

        for concept in expected_concepts:

            lexical_match = concept in response_concepts

            semantic_score = self.calculate_semantic_similarity(
                concept,
                response_sentences,
            )

            semantic_match = (
                semantic_score is not None
                and semantic_score >= self.semantic_threshold
            )

            if lexical_match or semantic_match:
                covered_concepts.add(concept)
            else:
                missing_concepts.add(concept)

        score = round(
            len(covered_concepts) / len(expected_concepts) * 100,
            2
        )

        if self.embedding_model is not None:
            scoring_method = "lexical and concept-level semantic matching"
        else:
            scoring_method = "lexical matching only"

        return {
            "judge_name": "Completeness Judge",
            "score": score,
            "passed": score >= self.completeness_threshold,
            "status": "evaluated",
            "explanation": (
                f"Completeness was estimated using {scoring_method}. "
                f"{len(covered_concepts)} of "
                f"{len(expected_concepts)} expected concepts "
                f"were identified as covered. "
                "Concept matching does not prove that every concept "
                "was adequately explained."
            ),
            "missing_concepts": sorted(missing_concepts),
            "covered_concepts": sorted(covered_concepts)
        }
