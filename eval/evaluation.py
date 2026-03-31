# simple_rag_generation_eval.py

import json

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric, 
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval
)

from rag.chat_rag import chat_rag




def build_judge_model():
    return OllamaModel(
        model="gemma3:1b",
        temperature=0,
    )


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

    judge_model = build_judge_model()

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

    answer_quality = GEval(
        name="Answer Quality",
        model=judge_model,
        threshold=0.7,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        criteria="""
        Evaluate the answer based on the following:
        1. Does it fully answer the question?
        2. Is it factually consistent with the provided context?
        3. Is any important information missing?
        4. Does it avoid making unsupported claims?

        The answer should be complete, correct, and grounded in the context.
        """,
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)
    answer_quality.measure(test_case)


    return {
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
        "answer_quality": {
            "score": answer_quality.score,
            "passed": answer_quality.success,
            "reason": answer_quality.reason,
        }
    }


def evaluate_retrieval(
    query: str,
    context: list[str],
    generated_answer: str,
    expected_answer: str | None = None,
    contextual_relevancy_threshold: float = 0.5,
    contextual_precision_threshold: float = 0.5,
    contextual_recall_threshold: float = 0.5,
):
    """
    Retriever evaluation.

    Parameters:
        query: user query
        context: retrieved chunks
        generated_answer: answer produced by your RAG generator
        expected_answer: ground-truth / reference answer
                         required for contextual precision and contextual recall
    """

    judge_model = build_judge_model()

    test_case = LLMTestCase(
        input=query,
        actual_output=generated_answer,
        expected_output=expected_answer,
        retrieval_context=context,
    )

    results = {
        "query": query,
        "retrieval_context": context,
    }

    contextual_relevancy = ContextualRelevancyMetric(
        model=judge_model,
        threshold=contextual_relevancy_threshold,
        include_reason=True,
    )
    contextual_relevancy.measure(test_case)

    results["contextual_relevancy"] = {
        "score": contextual_relevancy.score,
        "passed": contextual_relevancy.success,
        "reason": contextual_relevancy.reason,
    }

    # if expected_answer is not None:
    #     contextual_precision = ContextualPrecisionMetric(
    #         model=judge_model,
    #         threshold=contextual_precision_threshold,
    #         include_reason=True,
    #     )

    #     contextual_recall = ContextualRecallMetric(
    #         model=judge_model,
    #         threshold=contextual_recall_threshold,
    #         include_reason=True,
    #     )

    #     contextual_precision.measure(test_case)
    #     contextual_recall.measure(test_case)

    #     results["contextual_precision"] = {
    #         "score": contextual_precision.score,
    #         "passed": contextual_precision.success,
    #         "reason": contextual_precision.reason,
    #     }

    #     results["contextual_recall"] = {
    #         "score": contextual_recall.score,
    #         "passed": contextual_recall.success,
    #         "reason": contextual_recall.reason,
    #     }

    #     results["overall_passed"] = (
    #         contextual_relevancy.success
    #         and contextual_precision.success
    #         and contextual_recall.success
    #     )
    # else:
    # results["contextual_precision"] = None
    # results["contextual_recall"] = None
    # results["overall_passed"] = contextual_relevancy.success
    # results["note"] = (
    #     "expected_answer not provided, so only ContextualRelevancyMetric was run."
    # )

    return results

def evaluate_pipeline(
    query: str,
    context: list[str],
    generated_answer: str,
    expected_answer: str | None = None,
    contextual_relevancy_threshold: float = 0.5,
    contextual_precision_threshold: float = 0.5,
    contextual_recall_threshold: float = 0.5,
    answer_relevancy_threshold: float = 0.5,
    faithfulness_threshold: float = 0.5,
):
    result_gen = evaluate_generation(
        query=query,
        generated_answer=generated_answer,
        context=context
    )

    result_ret = evaluate_retrieval(
        query=query,
        generated_answer=generated_answer,
        context=context,
        expected_answer=expected_answer
    )

    
    # if expected_answer:
    #     return {
    #         "query": query,
    #         "generated_answer": generated_answer,
    #         "retrieved_context": context,
    #         "contextual_relevancy": {
    #             "score": result_ret.get("contextual_relevancy").get("score"),
    #             "passed": result_ret.get("contextual_relevancy").get("passed"),
    #             "reason": result_ret.get("contextual_relevancy").get("reason")
    #         },
    #         "contextual_recall": {
    #             "score": result_ret.get("contextual_recall", None).get("score"),
    #             "passed": result_ret.get("contextual_recall", None).get("passed"),
    #             "reason": result_ret.get("contextual_recall", None).get("reason")
    #         },
    #         "contextual_precision": {
    #             "score": result_ret.get("contextual_precision", None).get("score"),
    #             "passed": result_ret.get("contextual_precision", None).get("passed"),
    #             "reason": result_ret.get("contextual_precision", None).get("reason")
    #         },
    #         "answer_relevancy": {
    #             "score": result_gen.get("answer_relevancy").get("score"),
    #             "passed": result_gen.get("answer_relevancy").get("passed"),
    #             "reason": result_gen.get("answer_relevancy").get("reason"),
    #         },
    #         "faithfulness": {
    #             "score": result_gen.get("faithfulness").get("score"),
    #             "passed": result_gen.get("faithfulness").get("passed"),
    #             "reason": result_gen.get("faithfulness").get("reason"),
    #         },
    #         "answer_quality": {
    #             "score": result_gen.get("answer_quality").get("score"),
    #             "passed": result_gen.get("answer_quality").get("passed"),
    #             "reason": result_gen.get("answer_quality").get("reason"),
    #         }
    #     }
    # else:
    return {
        "query": query,
        "generated_answer": generated_answer,
        "retrieved_context": context,
        "contextual_relevancy": {
            "score": result_ret.get("contextual_relevancy").get("score"),
            "passed": result_ret.get("contextual_relevancy").get("passed"),
            "reason": result_ret.get("contextual_relevancy").get("reason")
        },
        "answer_relevancy": {
            "score": result_gen.get("answer_relevancy").get("score"),
            "passed": result_gen.get("answer_relevancy").get("passed"),
            "reason": result_gen.get("answer_relevancy").get("reason"),
        },
        "faithfulness": {
            "score": result_gen.get("faithfulness").get("score"),
            "passed": result_gen.get("faithfulness").get("passed"),
            "reason": result_gen.get("faithfulness").get("reason"),
        },
        "answer_quality": {
            "score": result_gen.get("answer_quality").get("score"),
            "passed": result_gen.get("answer_quality").get("passed"),
            "reason": result_gen.get("answer_quality").get("reason"),
        }
    }



if __name__ == "__main__":

    results = []
    dataset = []

    with open("C:/Users/banuv/Desktop/Talrop-GenAI_Avatar/eval/dataset.json", "r") as f:
        dataset = json.load(f)

    for sample in dataset:
        query = sample["query"]

        rag_result = chat_rag(
            org_id="82142c62-9820-4a30-95d6-2f337cc7c2e5",
            user_message=query,
            history=[],
            temperature=0,
            max_new_tokens=3000,
        )

        result = evaluate_pipeline(
            query=query,
            context=[rag_result.context],  # wrap as list[str]
            generated_answer=rag_result.answer,
            expected_answer=sample["expected_answer"]
        )

        results.append({
            "id": sample["id"],
            "query": sample["query"],
            "type": sample["type"],
            "difficulty": sample["difficulty"],
            "rag_answer": rag_result.answer,
            "result": result
        })

    with open("C:/Users/banuv/Desktop/Talrop-GenAI_Avatar/eval/results.json", "w") as f:
        json.dump(results, f, indent=2)


