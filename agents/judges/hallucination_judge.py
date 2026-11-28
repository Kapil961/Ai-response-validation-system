
import re
from typing import Optional

import numpy as np


class HallucinationJudge:
    def __init__(
        self,
        support_threshold: float = 0.70,
        embedding_model=None,
        semantic_threshold: float = 0.65,
        lexical_weight: float = 0.25,
        semantic_weight: float = 0.75,
    ):
        if not 0 <= support_threshold <= 1:
            raise ValueError("Support threshold must be between 0 and 1.")

        if not 0 <= semantic_threshold <= 1:
            raise ValueError("Semantic threshold must be between 0 and 1.")

        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError("Scoring weights cannot be negative.")

        if lexical_weight + semantic_weight == 0:
            raise ValueError("At least one scoring weight must be positive.")

        self.support_threshold = support_threshold
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

        self.stop_words = {
            "the", "is", "are", "was", "were", "a", "an",
            "and", "or", "but", "of", "in", "on", "at",
            "to", "for", "from", "by", "with", "as",
            "it", "this", "that", "these", "those",
            "he", "she", "they", "we", "you", "i",
            "has", "have", "had", "be", "been", "being"
        }

    def tokenize(self, text: str) -> set:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {
            word for word in words
            if word not in self.stop_words
        }


    def split_sentences(self, text: str) -> list:
        if not text or not text.strip():
            return []

        # Normalize line endings.
        cleaned_text = text.strip().replace("\r\n", "\n")

        # Remove numbered-list markers at the beginning of lines.
        cleaned_text = re.sub(
            r"(?m)^[ \t]*\d+[.)][ \t]*",
            "",
            cleaned_text
        )

        # Split headings ending with a colon onto separate segments.
        cleaned_text = re.sub(
            r":\s*\n+",
            ":\n",
            cleaned_text
        )

        # Split at newlines and sentence-ending punctuation.
        sentences = re.split(
            r"\n+|(?<!\d)(?<=[.!?])\s+",
            cleaned_text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
            and not sentence.strip().endswith(":")
        ]


    def has_negation(self, text: str) -> bool:
        return bool(
            re.search(r"\b(not|never|no)\b", text.lower())
        )

    def is_direct_contradiction(
        self,
        claim: str,
        source_document: str,
    ) -> bool:
        claim_negated = self.has_negation(claim)

        ignored_words = {
            "not", "never", "no", "do", "does", "did",
            "can", "could", "will", "would", "should"
        }

        def normalize(text):
            terms = self.tokenize(text) - ignored_words
            return {
                word[:-1]
                if word.endswith("s") and len(word) > 4
                else word
                for word in terms
            }

        claim_terms = normalize(claim)

        if not claim_terms:
            return False

        for source_sentence in self.split_sentences(source_document):
            source_negated = self.has_negation(source_sentence)

            if claim_negated == source_negated:
                continue

            source_terms = normalize(source_sentence)

            if not source_terms:
                continue

            overlap = len(claim_terms & source_terms)

            similarity = (
                overlap / min(len(claim_terms), len(source_terms))
            )

            if similarity >= 0.80:
                return True

        return False

    def calculate_lexical_score(
        self,
        claim: str,
        evidence: str,
    ) -> tuple:
        claim_words = self.tokenize(claim)
        evidence_words = self.tokenize(evidence)

        if not claim_words:
            return 0.0, []

        matched_words = claim_words.intersection(evidence_words)

        score = (
            len(matched_words) / len(claim_words)
        ) * 100

        return round(score, 2), sorted(matched_words)

    def calculate_semantic_similarity(
        self,
        claim: str,
        evidence_sentences: list,
    ) -> Optional[float]:
        if self.embedding_model is None or not evidence_sentences:
            return None

        try:
            texts = [claim] + evidence_sentences

            embeddings = self.embedding_model.encode(texts)

            similarities = np.dot(
                embeddings[1:],
                embeddings[0]
            )

            best_similarity = float(np.max(similarities))

            return round(
                max(0.0, min(1.0, best_similarity)) * 100,
                2
            )

        except Exception:
            return None



    def evaluate_claim(
        self,
        claim: str,
        source_document: str,
    ) -> dict:
        if self.is_direct_contradiction(claim, source_document):
            return {
                "claim": claim,
                "support_score": 0.0,
                "classification": "contradictory",
                "matched_evidence": None,
                "lexical_score": 0.0,
                "semantic_score": None,
                "matched_evidence_terms": []
            }

        source_sentences = self.split_sentences(source_document)

        if not source_sentences:
            return {
                "claim": claim,
                "support_score": 0.0,
                "classification": "potentially_unsupported",
                "matched_evidence": None,
                "lexical_score": 0.0,
                "semantic_score": None,
                "matched_evidence_terms": []
            }

        evidence_candidates = []

        for sentence in source_sentences:
            evidence_candidates.append(sentence)

            clauses = re.split(r"[,;]\s*", sentence)

            evidence_candidates.extend(
                clause.strip()
                for clause in clauses
                if clause.strip()
            )

        evidence_candidates = list(dict.fromkeys(evidence_candidates))

        best_support_score = -1.0
        best_lexical_score = 0.0
        best_semantic_score = None
        best_evidence = None
        best_matched_words = []

        for evidence in evidence_candidates:
            lexical_score, matched_words = (
                self.calculate_lexical_score(claim, evidence)
            )

            semantic_score = self.calculate_semantic_similarity(
                claim,
                [evidence]
            )

            if semantic_score is not None:
                total_weight = (
                    self.lexical_weight + self.semantic_weight
                )

                support_score = (
                    self.lexical_weight * lexical_score
                    + self.semantic_weight * semantic_score
                ) / total_weight
            else:
                support_score = lexical_score

            if support_score > best_support_score:
                best_support_score = support_score
                best_lexical_score = lexical_score
                best_semantic_score = semantic_score
                best_evidence = evidence
                best_matched_words = matched_words

        if best_semantic_score is not None:
            supported = (
                best_semantic_score >= self.semantic_threshold * 100
                and best_support_score >= self.support_threshold * 100
            )
        else:
            supported = (
                best_support_score >= self.support_threshold * 100
            )

        return {
            "claim": claim,
            "support_score": round(best_support_score, 2),
            "classification": (
                "supported"
                if supported
                else "potentially_unsupported"
            ),
            "matched_evidence": best_evidence,
            "lexical_score": round(best_lexical_score, 2),
            "semantic_score": (
                round(best_semantic_score, 2)
                if best_semantic_score is not None
                else None
            ),
            "matched_evidence_terms": best_matched_words
        }


    def evaluate(
        self,
        response: str,
        source_document: Optional[str] = None,
    ) -> dict:
        if not source_document or not source_document.strip():
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "No source document was provided. "
                    "Claims cannot be checked for evidence support."
                ),
                "claims": []
            }

        source_words = self.tokenize(source_document)

        if not source_words:
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "The source document contains no usable evidence."
                ),
                "claims": []
            }

        sentences = self.split_sentences(response)

        claims = []

        for sentence in sentences:
            if not self.tokenize(sentence):
                continue

            claims.append(
                self.evaluate_claim(sentence, source_document)
            )

        if not claims:
            return {
                "judge_name": "Hallucination Detection Agent",
                "score": None,
                "passed": None,
                "status": "insufficient_evidence",
                "explanation": (
                    "No evaluable claims were found in the response."
                ),
                "claims": []
            }

        supported_count = sum(
            claim["classification"] == "supported"
            for claim in claims
        )

        overall_score = round(
            (supported_count / len(claims)) * 100,
            2
        )

        return {
            "judge_name": "Hallucination Detection Agent",
            "score": overall_score,
            "passed": overall_score >= 70,
            "status": "evaluated",
            "explanation": (
                "This judge evaluates individual response claims "
                "against source sentences using lexical evidence "
                "matching and, when available, semantic similarity. "
                "It also applies a rule-based direct-negation check. "
                "These automated checks cannot guarantee factual "
                "verification or detect every contradiction."
            ),
            "claims": claims
        }
