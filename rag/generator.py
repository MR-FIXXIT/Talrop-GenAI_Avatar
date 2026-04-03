# rag/generator.py

from __future__ import annotations

import re
from typing import Dict, List, Literal, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace
from langchain_groq import ChatGroq

import rag.prompts as prompts


Role = Literal["user", "assistant"]
HistoryItem = Dict[str, str]

_NO_CONTEXT_ANSWER = "Answer: I don't know based on the provided context."

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_llm(*, temperature: float, max_new_tokens: int):
    # endpoint = HuggingFaceEndpoint(
    #     repo_id="Qwen/Qwen2.5-7B-Instruct",
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     top_p=1.0,
    #     repetition_penalty=1.05,
    #     do_sample=False if temperature == 0 else True,
    # )
    # return ChatHuggingFace(llm=endpoint)

    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=temperature,
        max_tokens=max_new_tokens,
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
    llm: ChatHuggingFace,
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
    llm: ChatHuggingFace,
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


def generate_answer_from_facts(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    question: str,
) -> str:
    """
    Generate an answer using a prompt that is expected to rely on supported facts.

    This function builds a simple system + human prompt chain and returns the model's
    generated answer after cleaning reasoning tags.

    Args:
        llm: The chat model used for answer generation.
        system_prompt: Prompt containing answer-generation instructions and/or facts.
        question: The user question passed as the human input.

    Returns:
        str: Generated answer after removing any <think> blocks.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"input": question})
    return strip_think(answer or "").strip()


def revise_answer_for_faithfulness(
    llm: ChatHuggingFace,
    *,
    system_prompt: str,
    question: str,
) -> str:
    """
    Revise a previously generated answer so it stays faithful to the provided context.

    This step is meant to improve grounding by re-checking and rewriting the answer
    according to a stricter system prompt. If the result is empty after cleanup,
    a fallback no-context answer is returned.

    Args:
        llm: The chat model used for revision.
        system_prompt: Prompt that instructs the model how to revise for faithfulness.
        question: The input passed to the revision prompt.

    Returns:
        str: Revised grounded answer, or a fallback answer if nothing valid is produced.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _escape_lc_braces(system_prompt)),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    revised = chain.invoke({"input": question})
    revised = strip_think(revised or "").strip()
    return revised or _NO_CONTEXT_ANSWER


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