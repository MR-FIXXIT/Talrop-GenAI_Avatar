# simple_rag_generation_eval.py

from deepeval.models import OllamaModel, GeminiModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase



def evaluate_generation(
    query: str,
    context: list[str],
    generated_answer: str,
    answer_relevancy_threshold: float = 0.5,
    faithfulness_threshold: float = 0.5,
):
    test_case = LLMTestCase(
        input=query,
        actual_output=generated_answer,
        retrieval_context=context,
    )

    judge_model = OllamaModel(
        model="gemma3:1b",
        temperature=0,
    )

    # judge_model = GeminiModel(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     api_key="AIzaSyDFUcNrRGaKhuel598dq8mjhVNkZZ38reA"
    # )

    answer_relevancy = AnswerRelevancyMetric(
        model=judge_model,
        threshold=answer_relevancy_threshold,
        include_reason=True,
        # verbose_mode=True
    )

    faithfulness = FaithfulnessMetric(
        model=judge_model,
        threshold=faithfulness_threshold,
        include_reason=True,
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)


    return {
        "query": query,
        "generated_answer": generated_answer,
        "answer_relevancy": {
            "score": answer_relevancy.score,
            "passed": answer_relevancy.success,
            "reason": answer_relevancy.reason,
        },
        "faithfulness": {
            "score": faithfulness.score,
            "passed": faithfulness.success,
            "reason": faithfulness.reason,
        },
        "overall_passed": answer_relevancy.success and faithfulness.success,
    }

