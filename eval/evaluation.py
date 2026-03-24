# simple_rag_generation_eval.py

import json

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric, 
    AnswerRelevancyMetric,
    FaithfulnessMetric
)


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

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)


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

    # 1) Referenceless retriever metric
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

    # 2) Reference-based retriever metrics
    if expected_answer is not None:
        contextual_precision = ContextualPrecisionMetric(
            model=judge_model,
            threshold=contextual_precision_threshold,
            include_reason=True,
        )

        contextual_recall = ContextualRecallMetric(
            model=judge_model,
            threshold=contextual_recall_threshold,
            include_reason=True,
        )

        contextual_precision.measure(test_case)
        contextual_recall.measure(test_case)

        results["contextual_precision"] = {
            "score": contextual_precision.score,
            "passed": contextual_precision.success,
            "reason": contextual_precision.reason,
        }

        results["contextual_recall"] = {
            "score": contextual_recall.score,
            "passed": contextual_recall.success,
            "reason": contextual_recall.reason,
        }

        results["overall_passed"] = (
            contextual_relevancy.success
            and contextual_precision.success
            and contextual_recall.success
        )
    else:
        results["contextual_precision"] = None
        results["contextual_recall"] = None
        results["overall_passed"] = contextual_relevancy.success
        results["note"] = (
            "expected_answer not provided, so only ContextualRelevancyMetric was run."
        )

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
        context=context
    )

    return {
        "query": query,
        "generated_answer": generated_answer,
        "retrieved_context": context,
        "contextual_relevancy": {
            "score": result_ret.get("contextual_relevancy").get("score"),
            "passed": result_ret.get("contextual_relevancy").get("passed"),
            "reason": result_ret.get("contextual_relevancy").get("reason")
        },
        # "contextual_recall": {
        #     "score": result_ret.get("contextual_recall", None).score,
        #     "passed": result_ret.get("contextual_recall", None).passed,
        #     "reason": result_ret.get("contextual_recall", None).reason
        # },
        # "contextual_precision": {
        #     "score": result_ret.get("contextual_precision", None).score,
        #     "passed": result_ret.get("contextual_precision", None).passed,
        #     "reason": result_ret.get("contextual_precision", None).reason
        # },
        "answer_relevancy": {
            "score": result_gen.get("answer_relevancy").get("score"),
            "passed": result_gen.get("answer_relevancy").get("passed"),
            "reason": result_gen.get("answer_relevancy").get("reason"),
        },
        "faithfulness": {
            "score": result_gen.get("faithfulness").get("score"),
            "passed": result_gen.get("faithfulness").get("passed"),
            "reason": result_gen.get("faithfulness").get("reason"),
        }
    }





if __name__ == "__main__":
    query = "Explain Fourier transform"

    context = [
        "[c1] (chunk_source=Dip textbook 4th edition.pdf, score=0.4378) 4.2 Preliminary Concepts 211 If f t( ) is real, we see that its transform in general is complex. Note that the Fourier transform is an expansion of f t( ) multiplied by sinusoidal terms whose frequencies are determined by the values of m. Thus, because the only variable left after integra- tion is frequency, we say that the domain of the Fourier transform is the frequency domain. We will discuss the frequency domain and its properties in more detail later in this chapter. In our discussion, t can represent any continuous variable, and the units of the frequency variable m depend on the units of t. For example, if t repre- sents time in seconds, the units of m are cycles/sec or Hertz (Hz). If t represents distance in meters, then the units of m are cycles/meter, and so on. In other words, the units of the frequency domain are cycles per unit of the independent variable of the input function. EXAMPLE 4.1 : Obtaining the Fourier transform of a simple continuous function. The Fourier transform of the function in Fig.4.4(a) follows from Eq. (4-20): F f t e dt Ae dt A j e j t j t j t W W ( ) ( ) m pm pm pm pm = = = − ⎡⎣ ⎤⎦ − − − − − -  2 2 2 2 2 2 2 2 W W j W j W j W j W A j e e A j e e AW 2 2 2 2 = − − ⎡⎣ ⎤⎦ = − ⎡⎣ ⎤⎦ = − − pm pm p pm pm pm pm sin( m pm W W ) ( ) t 0 f(t) A 0 W/2 W/2 0 1/W 1/W 1/W 1/W F(m) AW F(m) AW 2/W . . .", 
        "[c2] (chunk_source=Dip textbook 4th edition.pdf, score=0.4146) The formulation in this case is the Fourier transform, and its utility is even greater than the Fourier series in many theoretical and applied disciplines. Both representa- tions share the important characteristic that a function, expressed in either a Fourier series or transform, can be reconstructed (recovered) completely via an inverse pro- cess, with no loss of information. This is one of the most important characteristics of these representations because it allows us to work in the Fourier domain (generally called the frequency domain) and then return to the original domain of the function without losing any information. Ultimately, it is the utility of the Fourier series and transform in solving practical problems that makes them widely studied and used as fundamental tools. The initial application of Fourier’s ideas was in the ﬁeld of heat diffusion, where they allowed formulation of differential equations representing heat ﬂow in such a way that solutions could be obtained for the ﬁrst time. During the past century, and especially in the past 60 years, entire industries and academic disciplines have ﬂourished as a result of Fourier’s initial ideas. The advent of digital computers and the “discovery” of a fast Fourier transform (FFT) algorithm in the early 1960s revo- lutionized the ﬁeld of signal processing. These two core technologies allowed for the ﬁrst time practical processing of a host of signals of exceptional importance, ranging from medical monitors and scanners to modern electronic communications. As you learned in Section 3.4, it takes on the order of MNmn operations (multi- plications and additions) to ﬁlter an M N × image with a kernel of size m n × ele- ments. If the kernel is separable, the number of operations is reduced to MN m n ( ).", 
        "[c3] (chunk_source=Dip textbook 4th edition.pdf, score=0.4098) 204 Chapter 4 Filtering in the Frequency Domain 4.1 BACKGROUND We begin the discussion with a brief outline of the origins of the Fourier transform and its impact on countless branches of mathematics, science, and engineering. A BRIEF HISTORY OF THE FOURIER SERIES AND TRANSFORM The French mathematician Jean Baptiste Joseph Fourier was born in 1768 in the town of Auxerre, about midway between Paris and Dijon. The contribution for which he is most remembered was outlined in a memoir in 1807, and later pub- lished in 1822 in his book, La Théorie Analitique de la Chaleur (The Analytic Theory of Heat). This book was translated into English 55 years later by Freeman (see Freeman [1878]). Basically, Fourier’s contribution in this field states that any peri- odic function can be expressed as the sum of sines and/or cosines of different fre- quencies, each multiplied by a different coefficient (we now call this sum a Fourier series). It does not matter how complicated the function is; if it is periodic and satis- fies some mild mathematical conditions, it can be represented by such a sum. This is taken for granted now but, at the time it first appeared, the concept that compli- cated functions could be represented as a sum of simple sines and cosines was not at all intuitive (see Fig.4.1). Thus, it is not surprising that Fourier’s ideas were met initially with skepticism. Functions that are not periodic (but whose area under the curve is ﬁnite) can be expressed as the integral of sines and/or cosines multiplied by a weighting function."
    ]

    generated_answer = (
        "The Fourier transform is a mathematical tool that decomposes a function into a sum of sinusoidal terms with different frequencies. [c1] Explanation: If \( f(t) \) is a real function of time \( t \), its Fourier transform is generally a complex function that represents the contribution of different frequency components to the original function. [c1] These sinusoidal terms are determined by the values of \( m \), which represents the frequency. [c1] The domain of the Fourier transform is the frequency domain, meaning that the transformed function describes the frequency content of the original function. [c1]"
    )

    expected_answer = None

    results = evaluate_pipeline(query, context, generated_answer)

    with open("C:/Users/banuv/Desktop/Talrop-GenAI_Avatar/eval/retriever_result.json", "w") as f:
        json.dump(results, f, indent=2)
