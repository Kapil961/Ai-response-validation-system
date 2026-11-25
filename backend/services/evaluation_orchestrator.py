
from uuid import uuid4

from backend.models.evaluation import EvaluationResult, JudgeResult
from backend.models.submission import SubmissionRequest

from agents.judges.relevance_judge import RelevanceJudge
from agents.judges.accuracy_judge import AccuracyJudge
from agents.judges.hallucination_judge import HallucinationJudge
from agents.judges.completeness_judge import CompletenessJudge
from agents.verdict_agent import VerdictAgent


class EvaluationOrchestrator:

    def __init__(self, retrieval_engine=None):
        self.retrieval_engine = retrieval_engine

        self.relevance_judge = RelevanceJudge()

        # Reuse the embedding model already loaded by RetrievalEngine.
        self.accuracy_judge = AccuracyJudge(
            embedding_model=getattr(
                self.retrieval_engine,
                "embedding_model",
                None,
            )
        )

        self.hallucination_judge = HallucinationJudge(
    embedding_model=getattr(
        self.retrieval_engine,
        "embedding_model",
        None,
    )
)
        self.completeness_judge = CompletenessJudge()
        self.verdict_agent = VerdictAgent()

    def evaluate(self, submission: SubmissionRequest) -> EvaluationResult:

        # Retrieve supporting evidence only when no source is supplied.
        source_document = submission.source_document

        if not source_document and self.retrieval_engine is not None:
            retrieved_results = self.retrieval_engine.retrieve(
                submission.question
            )

            if retrieved_results:
                source_document = "\n\n".join(
                    result["text"] for result in retrieved_results
                )

        # Evaluate relevance.
        relevance = JudgeResult(
            **self.relevance_judge.evaluate(
                submission.question,
                submission.ai_response,
            )
        )

        # Accuracy continues to use the trusted reference answer.
        accuracy = JudgeResult(
            **self.accuracy_judge.evaluate(
                response=submission.ai_response,
                reference_answer=submission.reference_answer,
            )
        )

        # Evaluate faithfulness using supplied or retrieved evidence.
        faithfulness_data = self.hallucination_judge.evaluate(
            response=submission.ai_response,
            source_document=source_document,
        )

        claim_evidence = "; ".join(
            f'{claim["classification"]}: {claim["claim"]}'
            for claim in faithfulness_data.get("claims", [])
        ) or None

        faithfulness = JudgeResult(
            judge_name=faithfulness_data["judge_name"],
            score=faithfulness_data["score"],
            passed=faithfulness_data["passed"],
            explanation=faithfulness_data["explanation"],
            evidence=claim_evidence,
            status=faithfulness_data["status"],
        )

        # Evaluate completeness.
        completeness = JudgeResult(
            **self.completeness_judge.evaluate(
                question=submission.question,
                response=submission.ai_response,
            )
        )

        # Combine all judge results.
        judge_results = {
            "relevance": relevance,
            "accuracy": accuracy,
            "faithfulness": faithfulness,
            "completeness": completeness,
        }

        verdict_result = self.verdict_agent.evaluate(judge_results)

        return EvaluationResult(
            submission_id=str(uuid4()),
            overall_score=verdict_result["overall_score"],
            relevance=relevance,
            accuracy=accuracy,
            faithfulness=faithfulness,
            completeness=completeness,
            verdict=verdict_result["verdict"],
            summary=verdict_result["summary"],
        )
