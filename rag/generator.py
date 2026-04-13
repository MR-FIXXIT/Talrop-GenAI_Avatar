# rag/generator.py

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

import rag.prompts as prompts


HistoryItem = Dict[str, str]

_NO_CONTEXT_ANSWER = "Answer: I don't know based on the provided context."

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_llm(*, temperature: float, max_new_tokens: int, thinking: bool = False):
    """
    Build a ChatGroq LLM instance.

    Args:
        temperature:    Sampling temperature (0 = deterministic).
        max_new_tokens: Hard token budget for the response.
        thinking:       Whether to enable Qwen3's chain-of-thought reasoning.
                        Default False — reasoning chains can silently consume
                        thousands of tokens before the visible answer starts,
                        causing 35-50 s response times.
                        Set True only when deep reasoning is needed.
    """
    # reasoning_effort is a first-class ChatGroq field (not model_kwargs).
    # "none"    → disables Qwen3 <think> chains entirely (fastest)
    # "default" → enables standard chain-of-thought reasoning
    reasoning = "default" if thinking else "none"

    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=temperature,
        max_tokens=max_new_tokens,
        reasoning_effort=reasoning,
    )


def history_to_lc(history: Sequence[HistoryItem]) -> List[Tuple[str, str]]:
    # Convert [{"role": "user"/"assistant", "content": "..."}] -> [("human", "..."), ("ai", "...")].

    out: List[Tuple[str, str]] = []

    for m in history:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(("human", content) if role == "user" else ("ai", content))
    return out


def _escape_lc_braces(text: str) -> str:
    # Prevent LangChain from treating any {..} in retrieved context/org prompt as template variables
    return text.replace("{", "{{").replace("}", "}}")


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def contextualize_question(
    llm: ChatGroq,
    *,
    user_message: str,
    lc_history: List[Tuple[str, str]],
) -> str:
    """
    Rewrite the current user message into a standalone question using chat history.

    If prior conversation exists, this function asks the model to convert the
    latest user message into a self-contained question that can be understood
    without the previous chat context. This is useful before retrieval.

    If no history is provided, the original user message is returned as-is.

    Args:
        llm: The chat model used for rewriting.
        user_message: The user's latest message.
        lc_history: Conversation history already converted into LangChain format.

    Returns:
        str: A standalone question suitable for retrieval.
             Falls back to the original user message if rewriting fails or is empty.
    """

    if not lc_history:
        return user_message.strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.contextualize_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    q = chain.invoke({"chat_history": lc_history, "input": user_message})
    q = strip_think(q or "")
    return q.strip() or user_message.strip()


def extract_supported_facts(
    llm: ChatGroq,
    *,
    system_prompt: str,
    question: str,
) -> str:
    """
    Ask the model to extract only the facts supported by the provided prompt/context.

    This step is typically used in a grounded RAG pipeline to first pull out
    evidence-backed facts before composing a final answer.

    Args:
        llm: The chat model used for extraction.
        system_prompt: Instruction prompt containing the extraction rules and context.
        question: The user question or extraction input passed as the human message.

    Returns:
        str: Extracted supported facts after removing any <think> blocks.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    facts = chain.invoke({"input": question})
    return strip_think(facts or "").strip()




def normalize_final_answer(text: str) -> str:
    """
    Clean and normalize the final answer before returning it to the caller.

    This removes any <think> blocks, trims whitespace, and ensures that
    an empty result is replaced with a safe fallback answer.

    Args:
        text: Raw final answer text.

    Returns:
        str: Clean final answer ready to be returned to the user.
    """
    text = strip_think(text or "").strip()
    return text or _NO_CONTEXT_ANSWER


def generate_faithful_answer(
    llm: ChatGroq,
    *,
    system_prompt: str,
    question: str,
) -> str:
    """
    Generate a grounded, faithful answer in a single LLM call.

    This replaces the previous two-step pipeline of:
      1. generate_answer_from_facts  (~35 s)
      2. revise_answer_for_faithfulness (~49 s)

    By baking faithfulness constraints directly into the generation prompt,
    we eliminate one full LLM inference pass and cut answer latency in half
    with no loss in grounding quality.

    Args:
        llm:           The LLM used for generation.
        system_prompt: The faithful_answer_system prompt (facts + context +
                       faithfulness rules combined).
        question:      The standalone question passed as the human message.

    Returns:
        str: Final faithful answer ready to return to the user, or the
             no-context fallback if the result is empty.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"input": question})
    answer = strip_think(answer or "").strip()
    return answer or _NO_CONTEXT_ANSWER